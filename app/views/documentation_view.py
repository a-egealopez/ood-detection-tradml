"""Documentation view: project overview, architecture, methodology and references."""

import streamlit as st

CONCEPTS = [
    (
        "Anomaly types (point / contextual / collective)",
        (
            "A point anomaly is a single atypical event; contextual anomalies are only "
            "anomalous in a given context (e.g. an unusual hour); collective anomalies "
            "break the pattern of a whole sequence."
        ),
        (
            "Each detector targets a subset of these — the feature-extraction tutorial "
            "states which type each method can catch."
        ),
    ),
    (
        "Window aggregation features",
        (
            "Daily statistics over event counts (n_events, activity_hours, entropy_hourly, "
            "night_activity, ...) that summarize the hourly distribution of a day."
        ),
        (
            "The workhorse of activity recognition on binary sensors; the 9 features used "
            "by the pipeline live in `TemporalFeatureExtractor`."
        ),
    ),
    (
        "Inter-event interval (IEI), CV and Fano factor",
        (
            "The times between consecutive events viewed as a point process. The "
            "coefficient of variation (std / mean) and the Fano factor (variance / mean) "
            "distinguish regular, Poisson-like and bursty processes."
        ),
        (
            "A single gap flags a point anomaly; a day with an atypical rhythm flags a "
            "collective one."
        ),
    ),
    (
        "N-gram / Markov transition entropy",
        (
            "A first-order Markov chain over triggered sensors; the entropy of the "
            "transition matrix measures how predictable the event sequence is."
        ),
        (
            "Catches routine breaks even when every individual sensor and interval looks "
            "normal."
        ),
    ),
    (
        "Hidden Markov Model (HMM)",
        (
            "A latent-state model for sequential data; each state emits a Gaussian over "
            "the features and regime changes are scored."
        ),
        "Used in the sequential track to detect changes in activity regime across days.",
    ),
    (
        "Hawkes self-exciting process",
        (
            "A temporal point process where past events raise the instantaneous rate of "
            "future events (excitation with an exponential decay)."
        ),
        (
            "Models bursts of sensor activity; the current implementation is illustrative "
            "(scores the aggregated rhythm, not a full intensity fit)."
        ),
    ),
    (
        "AUROC",
        (
            "Area under the receiver-operating-characteristic curve: probability that a "
            "random anomaly scores higher than a random normal point (0.5 random, 1.0 "
            "perfect)."
        ),
        "Primary quality metric; above 0.75 is considered good for this domain.",
    ),
    (
        "Synthetic anomaly injection",
        (
            "Because the CASAS data has no real labels, anomalies are injected on the "
            "holdout split (±magnitude std on a subset of features) and detection is scored "
            "against those labels."
        ),
        (
            "A proxy for real anomalies — good for tuning, never a guarantee on production "
            "data."
        ),
    ),
]


def _render_concepts_references() -> None:
    st.markdown(
        "Each concept links to curated papers and forum threads. References are being "
        "added progressively."
    )
    for title, definition, why in CONCEPTS:
        st.markdown(f"**{title}**")
        st.markdown(f"*Definition:* {definition}")
        st.markdown(f"*Why it matters:* {why}")
        st.markdown("*References:* `[papers]` `[forums]` — to be filled per concept.")
        st.markdown("---")


def render_documentation_view() -> None:
    st.subheader("Documentation & Architecture")

    with st.expander("Project Overview", expanded=True):
        st.markdown(
            """
            This application is an **unsupervised anomaly detection system** for IoT
            sensor data (CASAS - Center for Advanced Studies in Adaptive Systems). It
            combines multiple detection algorithms in an ensemble for robustness, and
            offers a didactic track to understand how each algorithm draws its decision
            boundary on synthetic 2-D data.

            Two learning tracks are exposed through a guided workflow
            (**Data -> Features -> Detect**):
            - **2D Playground**: inspect each detector's real decision boundary on
              synthetic datasets (blobs, moons, circles, swiss roll).
            - **CASAS track**: ingest smart-home event streams, extract daily features
              and score them with an ensemble of vectorial and sequential detectors.
            """
        )

    with st.expander("Detector Types"):
        st.markdown(
            """
            **Vectorial (distance / density based)**:
            - **Isolation Forest**: random partitioning of the feature space.
            - **Extended IForest**: sliced paths for high-dimensional data.
            - **Mahalanobis**: distance metric that respects covariance.
            - **Elliptic Envelope**: robust Gaussian fitting (MCD).
            - **Robust Covariance**: Minimum Covariance Determinant.
            - **KNN**: distance to the k-th nearest neighbor.
            - **One-Class SVM**: non-convex boundary learning.
            - **LOF**: local density factor.
            - **Z-Score**: per-feature standard-deviation score.
            - **PCA Reconstruction**: reconstruction error to the dominant subspace.

            **Sequential (time series)**:
            - **HMM**: Hidden Markov Model transitions.
            - **Hawkes**: self-exciting point process.
            """
        )

    with st.expander("Feature Extraction"):
        st.markdown(
            """
            Three methods from the event-driven time-series literature:
            1. **Window Aggregation** (contextual + collective): per-window statistics
               (counts, entropy, night share); detects atypical days as a whole.
            2. **Inter-Event Interval (IEI)** (point / collective): analyzes the time
               between events as a point process; CV and Fano factor capture
               regularity / burstiness.
            3. **N-gram Transition / Markov** (collective sequence): first-order
               transition matrix of sensors; detects routines that break the usual
               sequence pattern.
            """
        )

    with st.expander("Key Metrics"):
        st.markdown(
            """
            - **Precision**: TP / (TP + FP)
            - **Recall**: TP / (TP + FN)
            - **F1-Score**: harmonic mean of precision and recall.
            - **AUROC**: area under the ROC curve (0.5 random, 1.0 perfect).
            """
        )

    with st.expander("Ensemble Strategies"):
        st.markdown(
            """
            **Soft voting (weighted sum rule)**:
            $$S_{ensemble}(x) = \\sum_{i=1}^{n} w_i \\cdot s_i(x)$$
            - *Uniform*: all detectors are equally important.
            - *Entropy-based*: confident detectors get higher weight.

            **Hard voting (majority rule)**: each detector emits a binary vote;
            robust to outliers in individual scores, but loses the continuous-score
            information.
            """
        )

    with st.expander("Evaluation Without Labels"):
        st.markdown(
            """
            Since the CASAS datasets lack real anomaly labels, evaluation uses
            **synthetic anomaly injection**:
            1. Train on ~70% of the daily features.
            2. Keep ~30% as holdout.
            3. Inject synthetic anomalies (±`magnitude` std on a subset of features, on
               ~`contamination` of the holdout).
            4. Compute Precision, Recall, F1 and AUROC against the synthetic labels.

            **Limitation**: real anomalies may differ from synthetic patterns. This is a
            proxy evaluation used to tune hyperparameters.
            """
        )

    with st.expander("Concepts & References", expanded=False):
        _render_concepts_references()
