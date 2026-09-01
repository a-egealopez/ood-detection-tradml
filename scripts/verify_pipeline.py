"""End-to-end verification of the coherent-anomaly pipeline.

Runs every verification gate and exits non-zero if any fails:

 1. Generator gates   : graph entropy < 0.70, AC1(n_events) >= 0.30
                        (``tests/unit/test_markov_generator_unit.py``).
 2. Injector gates    : marginal preservation (contextual: total+sensor constant,
                        hourly changed; collective: total+sensor+hour constant)
                        and monotonic intensity proxies
                        (``tests/unit/test_injectors_unit.py``).
 3. Detector gates    : HMM predictive (regime change > normal), Hawkes intensity
                        (burst / displaced day > baseline), MarkovSequence
                        (reversed day > typical) — the behavioral tests under
                        ``tests/functional/``, run via pytest.
 4. Matrix gates      : the type x intensity x detector matrix on the houses
                        (``evaluation.matrix_evaluation``). All three types are
                        injected on the raw event stream (intensity-graded):
                          - point      -> distance family AUROC >= 0.85 at high
                                          intensity, and rises monotonically
                          - contextual -> Z-Score/HMM/Hawkes >= 0.75 at high
                                          intensity, Markov Sequence ~ 0.5 (blind)
                          - collective -> Markov Sequence >= 0.75, distance/HMM/
                                          Hawkes <= 0.65 (blind)
                          - control    -> null control ~ 0.5 for every detector
                          - expected winners rise monotonically low->high

The collective gate is source-aware: it is a hard gate only when the stream's
transitions are directional (synthetic houses; asymmetry < 0.85). On near-symmetric
data (real homes) order-reversal produces no rare transitions by construction, so
the gate is reported as informational instead of failing.

Usage:  python scripts/verify_pipeline.py [--n-seeds 5] [--source synthetic]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import db_path, setup_logging  # noqa: E402
from detectors.constants import DEFAULT_RANDOM_STATE  # noqa: E402
from evaluation.event_injection import transition_asymmetry  # noqa: E402
from evaluation.matrix_evaluation import (  # noqa: E402
    DEFAULT_DETECTORS,
    aggregate_matrix,
    monotonicity_check,
    run_matrix,
)
from features import NextEventTransitionExtractor  # noqa: E402
from ingestion.sqlite_manager import SQLiteDataManager  # noqa: E402

logger = setup_logging()

MATRIX_WINNERS = {
    "point": ["Z-Score", "Mahalanobis", "Isolation Forest", "PCA Reconstruction"],
    "contextual": ["Z-Score", "HMM", "Hawkes"],
    "collective": ["Markov Sequence"],
}
BLIND_ON_COLLECTIVE = [
    "Z-Score", "Mahalanobis", "Isolation Forest", "PCA Reconstruction", "HMM", "Hawkes",
]

GATE_WINNER_POINT = 0.85
GATE_WINNER_GRADED = 0.75
GATE_BLIND_COLLECTIVE = 0.65
GATE_NULL_TOL = 0.12      # null-control AUROC must stay within [0.38, 0.62]
GATE_MARKOV_CONTEXTUAL = 0.15  # |auroc - 0.5| must stay under this
GATE_ASYMMETRY = 0.85          # transitions must be directional for reversal work

# The generator/injector/detector gates live as pytest tests under tests/;
# verify_pipeline runs them in one subprocess call instead of invoking each
# module's __main__ block.
FASE123_TESTS = [
    "tests/unit/test_markov_generator_unit.py",
    "tests/unit/test_injectors_unit.py",
    "tests/functional/test_hmm_behavior.py",
    "tests/functional/test_hawkes_behavior.py",
    "tests/functional/test_markov_sequence_behavior.py",
]


def compute_asymmetry(houses: dict[str, pd.DataFrame]) -> float:
    """Mean transition asymmetry over houses (1.0 = symmetric, <1.0 = directional)."""
    ratios = []
    for df_house in houses.values():
        extractor = NextEventTransitionExtractor()
        extractor.fit(df_house)
        ratios.append(transition_asymmetry(df_house, extractor))
    return float(np.mean(ratios)) if ratios else 1.0


def run_pytest_tests(paths: list[str]) -> tuple[bool, str]:
    """Run a set of pytest files (exit 0 = gates passed).

    Runs with the project root on PYTHONPATH so pytest discovers ``src`` and the
    shared ``tests/conftest.py`` fixtures.
    """
    env = {
        "PYTHONPATH": str(ROOT),
        "PYTHONWARNINGS": "ignore",
    }
    # `where` is a fixed constant; the env dict is built from known values, so a
    # subprocess call here does not execute untrusted input.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-q", *paths],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    tail = proc.stdout.strip()
    if not tail and proc.returncode != 0:
        tail = proc.stderr.strip()
    return proc.returncode == 0, tail


def add(name: str, ok: bool, detail: str, checks: list) -> None:
    label = "OK" if ok else "FAIL"
    checks.append((name, label, detail))
    print(f"  [{label}] {name}: {detail}")


def check_matrix_gates(agg, checks, *, asymmetry_ok: bool) -> None:
    def get(atype: str, intensity: str, detector: str):
        rows = agg[
            (agg["anomaly_type"] == atype)
            & (agg["intensity"] == intensity)
            & (agg["detector"] == detector)
        ]
        return float(rows["auroc_mean"].iloc[0]) if len(rows) else None

    def add(name: str, ok: bool, detail: str, *, info: bool = False) -> None:
        label = "INFO" if info else ("OK" if ok else "FAIL")
        checks.append((name, label, detail))
        print(f"  [{label}] {name}: {detail}")

    for det in MATRIX_WINNERS["point"]:
        v = get("point", "high", det)
        add(f"point/{det} >= {GATE_WINNER_POINT}",
            v is not None and v >= GATE_WINNER_POINT, f"AUROC={v:.3f}")

    for det in MATRIX_WINNERS["contextual"]:
        v = get("contextual", "high", det)
        add(f"contextual/{det} >= {GATE_WINNER_GRADED}",
            v is not None and v >= GATE_WINNER_GRADED, f"AUROC={v:.3f}")
    v = get("contextual", "high", "Markov Sequence")
    add("contextual/Markov Sequence ~ 0.5 (blind)",
        v is not None and abs(v - 0.5) <= GATE_MARKOV_CONTEXTUAL, f"AUROC={v:.3f}")

    v = get("collective", "high", "Markov Sequence")
    if asymmetry_ok:
        add(f"collective/Markov Sequence >= {GATE_WINNER_GRADED}",
            v is not None and v >= GATE_WINNER_GRADED, f"AUROC={v:.3f}")
    else:
        add(f"collective/Markov Sequence >= {GATE_WINNER_GRADED}", True,
            f"AUROC={v:.3f} — INFO: symmetric transitions; reversal undetectable by design",
            info=True)
    for det in BLIND_ON_COLLECTIVE:
        v = get("collective", "high", det)
        add(f"collective/{det} <= {GATE_BLIND_COLLECTIVE} (blind)",
            v is not None and v <= GATE_BLIND_COLLECTIVE, f"AUROC={v:.3f}")

    for det in DEFAULT_DETECTORS:
        v = get("control", "none", det)
        add(f"control/{det} ~ 0.5",
            v is not None and abs(v - 0.5) <= GATE_NULL_TOL, f"AUROC={v:.3f}")

    mono = monotonicity_check(agg, MATRIX_WINNERS)
    for key, (ok, values) in mono.items():
        if ok is None:
            continue
        if "collective" in key and not asymmetry_ok:
            add(f"monotonicity/{key}", True, f"{values} — INFO: symmetric transitions",
                info=True)
        else:
            add(f"monotonicity/{key}", ok, str(values))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    parser.add_argument("--houses", nargs="+", default=None)
    args = parser.parse_args()

    checks: list[tuple[str, str, str]] = []
    print("== Generator / injector / detector gates (pytest) ==")
    ok, out = run_pytest_tests(FASE123_TESTS)
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    add("Self-validation (generator + injectors + detectors)", ok, tail, checks)

    print("\n== Matrix gates ==")
    db = SQLiteDataManager(str(db_path(args.source)))
    db.connect()
    houses = args.houses or db.list_houses()
    if not houses:
        print("Database empty; run scripts/generate_test_fixtures.py and casas_loader.py first.")
        db.close()
        sys.exit(1)

    frames = []
    house_dfs = {}
    for house_id in houses:
        df_house = db.query_house(house_id)
        if df_house.empty:
            continue
        house_dfs[house_id] = df_house
        matrix = run_matrix(df_house, house_id, n_seeds=args.n_seeds, seed_base=args.seed_base)
        frames.append(matrix)
        print(f"  [{house_id}] matrix done ({len(matrix)} cells)")
    db.close()

    if not frames:
        print("No data to evaluate.")
        sys.exit(1)

    # The collective (order-reversal) gate only applies to directional data. On
    # near-symmetric streams (transition_asymmetry >= GATE_ASYMMETRY) reversal
    # produces no rare transitions and the gate would fail by construction — report
    # it as informational instead of failing. The synthetic houses are directional
    # (~0.35); the real CASAS homes measure 0.68-0.83 (below the threshold).
    asymmetry = compute_asymmetry(house_dfs)
    asymmetry_ok = asymmetry < GATE_ASYMMETRY
    print(f"  Transition asymmetry (1.0 = symmetric, gate < {GATE_ASYMMETRY}): {asymmetry:.3f} "
          f"-> collective gate {'HARD' if asymmetry_ok else 'informational (symmetric data)'}")

    agg = aggregate_matrix(pd.concat(frames, ignore_index=True))
    check_matrix_gates(agg, checks, asymmetry_ok=asymmetry_ok)

    print("\n== Summary ==")
    n_ok = sum(1 for _, label, _ in checks if label in ("OK", "INFO"))
    for name, label, _detail in checks:
        print(f"  [{label}] {name}")
    failed = [name for name, label, _ in checks if label == "FAIL"]
    print(f"\n{len(checks)} gates: {n_ok} passed, {len(failed)} failed.")
    if failed:
        print("Failed gates:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    print("All gates passed.")


if __name__ == "__main__":
    main()
