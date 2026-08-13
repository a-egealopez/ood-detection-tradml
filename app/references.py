"""Curated formal references: papers, surveys, library manuals and monographs.

Single source for the "Learn more" links shown on the detector cards (2D Playground)
and for the per-concept references + study guide in the Documentation tab. Every
resource is a primary paper, a peer-reviewed survey, an official library manual or
a monograph — no blog posts or wiki pages. Every URL was checked before being added.
"""

from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Resource model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Resource:
    """One curated reference for a detector or a concept.

    ``method`` is the key that links the resource back to a detector name (the
    ``DETECTOR_REGISTRY`` display names) or to a concept title (the Documentation
    tab). ``kind`` selects the badge shown in the UI.
    """

    method: str
    title: str
    kind: str  # paper | survey | manual | tool
    url: str
    note: str  # short didactic "why read this"


@dataclass(frozen=True)
class StudyStep:
    """One stage of the didactic study path."""

    stage: str
    title: str
    goal: str  # what the learner should be able to do after this stage
    topics: tuple[str, ...] = ()  # RESOURCES keys to link from this stage


# Badge shown next to each resource kind.
KIND_LABELS = {
    "paper": "📄 Paper",
    "survey": "📚 Survey",
    "manual": "📘 Docs",
    "tool": "🧰 Library",
}

# Concept titles that map onto a detector key (same references, two entry points).
ALIASES = {
    "Hidden Markov Model (HMM)": "HMM",
    "Hawkes self-exciting process": "Hawkes",
    "OC-SVM (RBF)": "OC-SVM",
    "OC-SVM (Linear)": "OC-SVM",
    "OC-SVM (Poly)": "OC-SVM",
}


# ----------------------------------------------------------------------------
# The registry: keyed by detector name or concept title
# ----------------------------------------------------------------------------
RESOURCES: tuple[Resource, ...] = (
    # --- 2D Playground: vectorial detectors --------------------------------
    Resource(
        method="Isolation Forest",
        title="Isolation Forest",
        kind="paper",
        url="https://www.lamda.nju.edu.cn/publication/icdm08b.pdf",
        note="Liu, Ting & Zhou (2008), IEEE ICDM: anomalies are rare and easy to isolate — random partitions alone, no distance model needed.",
    ),
    Resource(
        method="Isolation Forest",
        title="Isolation Forest (2012, TKDD)",
        kind="paper",
        url="https://dl.acm.org/doi/10.1145/2133360.2133363",
        note="Liu, Ting & Zhou (2012), ACM TKDD 6(3): journal version with the full theoretical analysis and further experiments.",
    ),
    Resource(
        method="Isolation Forest",
        title="IsolationForest example (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html",
        note="Official scikit-learn example: decision boundary and path-length score in 2-D — the same idea this Playground draws.",
    ),
    Resource(
        method="Isolation Forest",
        title="Novelty and Outlier Detection (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/modules/outlier_detection.html",
        note="Official reference: where the forest sits among the classical detectors and how to use it.",
    ),
    Resource(
        method="Extended IForest",
        title="Extended Isolation Forest",
        kind="paper",
        url="https://arxiv.org/abs/1811.02141",
        note="Hariri, Kind & Brunner (2019), IEEE TKDE: oblique (sliced) splits remove the axis-parallel bias — the 'ghost clusters' of the classic forest.",
    ),
    Resource(
        method="LOF",
        title="LOF: Identifying Density-Based Local Outliers",
        kind="paper",
        url="http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf",
        note="Breunig, Kriegel, Ng & Sander (2000), ACM SIGMOD: local vs. neighbour density — catches outliers global methods miss.",
    ),
    Resource(
        method="LOF",
        title="LoOP: Local Outlier Probabilities",
        kind="paper",
        url="https://doi.org/10.1145/1645953.1646195",
        note="Kriegel, Kröger, Schubert & Zimek (2009), ACM CIKM: normalizes the LOF score into [0, 1] probabilities — the same scale this app's scores use.",
    ),
    Resource(
        method="Mahalanobis",
        title="On the Generalised Distance in Statistics (1936)",
        kind="paper",
        url="https://link.springer.com/article/10.1007/s13171-019-00164-5",
        note="Mahalanobis (1936), Proc. National Institute of Sciences of India: the original covariance-aware distance, in an open-access reprint.",
    ),
    Resource(
        method="Mahalanobis",
        title="Robust covariance estimation and Mahalanobis distances (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html",
        note="Official scikit-learn example: how outliers contaminate the distance — the reason the robust detectors exist.",
    ),
    Resource(
        method="Mahalanobis",
        title="Minimum Covariance Determinant and Extensions",
        kind="paper",
        url="https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd.pdf",
        note="Hubert, Debruyne & Rousseeuw (2018), WIREs Computational Statistics: the didactic review that ties the whole Gaussian family together.",
    ),
    Resource(
        method="Elliptic Envelope",
        title="A Fast Algorithm for the Minimum Covariance Determinant Estimator",
        kind="paper",
        url="https://www.tandfonline.com/doi/abs/10.1080/00401706.1999.10485670",
        note="Rousseeuw & Van Driessen (1999), Technometrics 41(3): the FastMCD algorithm behind the robust Gaussian fit.",
    ),
    Resource(
        method="Elliptic Envelope",
        title="Minimum Covariance Determinant and Extensions",
        kind="paper",
        url="https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd.pdf",
        note="Hubert, Debruyne & Rousseeuw (2018), WIREs Computational Statistics: when the MCD ellipse is a good model — and when not.",
    ),
    Resource(
        method="Elliptic Envelope",
        title="Novelty and Outlier Detection (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/modules/outlier_detection.html",
        note="Official reference: assumes (approximately) Gaussian data; degrades on multimodal or heavily non-Gaussian distributions.",
    ),
    Resource(
        method="Robust Covariance",
        title="A Fast Algorithm for the Minimum Covariance Determinant Estimator",
        kind="paper",
        url="https://www.tandfonline.com/doi/abs/10.1080/00401706.1999.10485670",
        note="Rousseeuw & Van Driessen (1999), Technometrics 41(3): the algorithm this detector wraps.",
    ),
    Resource(
        method="Robust Covariance",
        title="Minimum Covariance Determinant and Extensions",
        kind="paper",
        url="https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd.pdf",
        note="Hubert, Debruyne & Rousseeuw (2018), WIREs Computational Statistics: why a high-breakdown covariance beats the empirical one.",
    ),
    Resource(
        method="Robust Covariance",
        title="Novelty and Outlier Detection (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/modules/outlier_detection.html",
        note="Official reference: Gaussian assumption — works well on elliptical data, fails on multimodal distributions.",
    ),
    Resource(
        method="KNN",
        title="Efficient Algorithms for Mining Outliers from Large Data Sets",
        kind="paper",
        url="https://webdocs.cs.ualberta.ca/~zaiane/pub/check/ramaswamy.pdf",
        note="Ramaswamy, Rastogi & Shim (2000), ACM SIGMOD: outlierness as the distance to the k-th neighbour.",
    ),
    Resource(
        method="KNN",
        title="PyOD: KNN model docs",
        kind="tool",
        url="https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.knn",
        note="Reference implementation with the mean / largest / median variants of the k-NN score.",
    ),
    Resource(
        method="OC-SVM",
        title="Support Vector Method for Novelty Detection",
        kind="paper",
        url="https://papers.nips.cc/paper/1723-support-vector-method-for-novelty-detection.pdf",
        note="Schölkopf, Williamson, Smola, Shawe-Taylor & Platt (2000), NIPS 12: the kernel boundary and the meaning of nu.",
    ),
    Resource(
        method="OC-SVM",
        title="Novelty and Outlier Detection (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/modules/outlier_detection.html",
        note="Official reference: OC-SVM suits novelty detection with clean training data — outliers in the training set skew it.",
    ),
    Resource(
        method="Z-Score",
        title="Detection of Outliers (NIST e-Handbook, §1.3.5.17)",
        kind="manual",
        url="https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm",
        note="The classic 3-sigma rule plus formal tests (Grubbs, ESD) and the masking problem a naive z-score ignores.",
    ),
    Resource(
        method="Z-Score",
        title="Anomaly Detection: A Survey",
        kind="survey",
        url="https://dl.acm.org/doi/10.1145/1541880.1541882",
        note="Chandola, Banerjee & Kumar (2009), ACM Computing Surveys: statistical (univariate) detection is the baseline every modern method builds on.",
    ),
    Resource(
        method="PCA Reconstruction",
        title="Pattern Recognition and Machine Learning — Ch. 12 (Bishop, 2006)",
        kind="manual",
        url="https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
        note="Bishop (2006): PCA as the dominant subspace; reconstruction error is its natural anomaly score.",
    ),
    Resource(
        method="PCA Reconstruction",
        title="Anomaly Detection: A Survey",
        kind="survey",
        url="https://dl.acm.org/doi/10.1145/1541880.1541882",
        note="Chandola, Banerjee & Kumar (2009), ACM Computing Surveys: situates reconstruction-error detection inside the broader taxonomy.",
    ),
    # --- Sequential detectors ---------------------------------------------
    Resource(
        method="HMM",
        title="A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition",
        kind="paper",
        url="https://web.mit.edu/6.435/www/Rabiner89.pdf",
        note="Rabiner (1989), Proc. IEEE 77(2): the canonical introduction to states, emissions and the forward/backward machinery — 20k+ citations.",
    ),
    Resource(
        method="HMM",
        title="hmmlearn",
        kind="tool",
        url="https://hmmlearn.readthedocs.io/",
        note="The library used in the sequential track; its tutorial maps 1:1 onto this detector.",
    ),
    Resource(
        method="Hawkes",
        title="Spectra of Some Self-Exciting and Mutually Exciting Point Processes",
        kind="paper",
        url="https://www.dcscience.net/Hawkes-Biometrika-1971.pdf",
        note="Hawkes (1971), Biometrika 58(1): the founding model — past events raise the rate of future ones.",
    ),
    Resource(
        method="Hawkes",
        title="A Tutorial on Hawkes Processes for Events in Social Media",
        kind="paper",
        url="https://arxiv.org/abs/1708.06401",
        note="Rizoiu, Lee, Mishra & Xie (2017), arXiv:1708.06401: the most accessible modern introduction to self-exciting point processes.",
    ),
    Resource(
        method="Hawkes",
        title="tick",
        kind="tool",
        url="https://x-datainitiative.github.io/tick/",
        note="The point-process library used here; simulates and fits Hawkes kernels (heavy native build).",
    ),
    # --- Concepts (Documentation tab) --------------------------------------
    Resource(
        method="Anomaly types (point / contextual / collective)",
        title="Anomaly Detection: A Survey",
        kind="survey",
        url="https://dl.acm.org/doi/10.1145/1541880.1541882",
        note="Chandola, Banerjee & Kumar (2009): the reference taxonomy and the assumptions behind each detector family.",
    ),
    Resource(
        method="Anomaly types (point / contextual / collective)",
        title="Comparing anomaly detection algorithms on toy datasets (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html",
        note="Official scikit-learn example: Robust Covariance, OC-SVM, Isolation Forest and LOF side-by-side on the same 2-D geometries.",
    ),
    Resource(
        method="Anomaly types (point / contextual / collective)",
        title="Novelty and Outlier Detection (scikit-learn)",
        kind="manual",
        url="https://scikit-learn.org/stable/modules/outlier_detection.html",
        note="Hands-on overview of the classical detectors this app wraps.",
    ),
    Resource(
        method="Anomaly types (point / contextual / collective)",
        title="PyOD: a Python toolbox for scalable outlier detection",
        kind="paper",
        url="https://jmlr.org/papers/volume20/19-011/19-011.pdf",
        note="Zhao, Nasrullah & Li (2019), JMLR: the library that covers almost every classical detector with one API.",
    ),
    Resource(
        method="Anomaly types (point / contextual / collective)",
        title="PyOD",
        kind="tool",
        url="https://pyod.readthedocs.io/",
        note="PyOD: dozens of classical detectors with one API — a great catalogue to compare families.",
    ),
    Resource(
        method="Ensemble detectors",
        title="Theoretical Foundations and Algorithms for Outlier Ensembles",
        kind="paper",
        url="https://charuaggarwal.net/theory.pdf",
        note="Aggarwal & Sathe (2015), ACM SIGKDD Explorations 17(1): why combining detectors works and how to combine their scores.",
    ),
    Resource(
        method="Ensemble detectors",
        title="Outlier Ensembles: An Introduction (Springer)",
        kind="manual",
        url="https://link.springer.com/book/10.1007/978-3-319-54765-7",
        note="Aggarwal & Sathe (2017): the book-length treatment of ensemble theory, base detectors and combination functions.",
    ),
    Resource(
        method="Ensemble detectors",
        title="PyOD model combination",
        kind="tool",
        url="https://pyod.readthedocs.io/",
        note="Practical score-combination recipes (average / maximization / AOM) used in real ensembles.",
    ),
    Resource(
        method="Window aggregation features",
        title="Human activity recognition in smart homes with binary sensors: a survey",
        kind="survey",
        url="https://www.sciencedirect.com/science/article/abs/pii/S1566253524005098",
        note="Recent (2024) survey; daily feature aggregation is the workhorse of HAR on binary sensors.",
    ),
    Resource(
        method="Window aggregation features",
        title="Learning Setting-Generalized Activity Models for Smart Spaces",
        kind="paper",
        url="https://eecs.wsu.edu/~cook/pubs/computer12.pdf",
        note="Cook (2012): how CASAS homes are instrumented and how daily activity is learned from the same features.",
    ),
    Resource(
        method="Inter-event interval (IEI), CV and Fano factor",
        title="The Statistical Analysis of Series of Events (Cox & Lewis)",
        kind="manual",
        url="https://link.springer.com/book/9789401178037",
        note="The classic reference on point processes; CV and Fano factor come straight from this tradition.",
    ),
    Resource(
        method="N-gram / Markov transition entropy",
        title="A Mathematical Theory of Communication (Shannon, 1948)",
        kind="paper",
        url="https://web.archive.org/web/20080516051043/cm.bell-labs.com/cm/ms/what/shannonday/paper.html",
        note="Shannon's entropy and why n-grams / Markov chains quantify predictability — the theory behind this feature.",
    ),
    Resource(
        method="AUROC",
        title="An Introduction to ROC Analysis (Fawcett, 2006)",
        kind="paper",
        url="https://math.ucdavis.edu/~saito/data/roc/fawcett-roc.pdf",
        note="Fawcett (2006), Pattern Recognition Letters 27(8): why ROC curves are the right tool when class balance is unknown — the metric behind the AUROC pill.",
    ),
    Resource(
        method="Synthetic anomaly injection",
        title="Anomaly Detection: A Survey",
        kind="survey",
        url="https://dl.acm.org/doi/10.1145/1541880.1541882",
        note="Why evaluation without labels relies on proxied / synthetic anomalies.",
    ),
    Resource(
        method="Synthetic anomaly injection",
        title="Synthetic Anomaly Test Sets from CASAS ARUBA (Zenodo)",
        kind="tool",
        url="https://zenodo.org/records/15800764",
        note="A research artifact that injects synthetic anomalies into CASAS Aruba — exactly the strategy used here.",
    ),
    # --- CASAS (study guide) ------------------------------------------------
    Resource(
        method="CASAS",
        title="Learning Setting-Generalized Activity Models for Smart Spaces",
        kind="paper",
        url="https://eecs.wsu.edu/~cook/pubs/computer12.pdf",
        note="How CASAS homes are instrumented and why the activity model matters — context for the whole CASAS track.",
    ),
    Resource(
        method="CASAS",
        title="CASAS Datasets",
        kind="tool",
        url="https://casas.wsu.edu/datasets/",
        note="The public dataset page; aruba, cairo, milan and tulum are used here.",
    ),
    Resource(
        method="CASAS",
        title="CASAS smart-home dataset (aruba, cairo, milan, tulum) on Zenodo",
        kind="tool",
        url="https://zenodo.org/records/17180309",
        note="A maintained mirror with floorplans and sensor maps for the four houses.",
    ),
)


def resources_for(method: str) -> tuple[Resource, ...]:
    """Curated references for a detector name or concept (alias-resolved)."""
    key = ALIASES.get(method, method)
    return tuple(res for res in RESOURCES if res.method == key)


# ----------------------------------------------------------------------------
# Study guide: a didactic path through the whole stack
# ----------------------------------------------------------------------------
STUDY_GUIDE: tuple[StudyStep, ...] = (
    StudyStep(
        stage="1",
        title="Univariate statistics: the z-score",
        goal="Read a z-score, know mean / std, and see why naive distance-from-the-mean breaks.",
        topics=("Z-Score",),
    ),
    StudyStep(
        stage="2",
        title="The anomaly taxonomy",
        goal="Classify anomalies as point / contextual / collective and map each to a detector family.",
        topics=("Anomaly types (point / contextual / collective)",),
    ),
    StudyStep(
        stage="3",
        title="Distance and covariance in multivariate space",
        goal="Understand Mahalanobis distance and the effect of an ill-estimated covariance.",
        topics=("Mahalanobis", "Elliptic Envelope", "Robust Covariance"),
    ),
    StudyStep(
        stage="4",
        title="Local density and neighbours",
        goal="Distinguish local from global outliers; know when KNN / LOF beat global methods.",
        topics=("KNN", "LOF"),
    ),
    StudyStep(
        stage="5",
        title="Non-parametric boundaries",
        goal="Wrap the normal region with a boundary instead of fitting a density.",
        topics=("OC-SVM",),
    ),
    StudyStep(
        stage="6",
        title="Low-dimensional structure",
        goal="Use reconstruction error to the dominant subspace as an anomaly score.",
        topics=("PCA Reconstruction",),
    ),
    StudyStep(
        stage="7",
        title="Isolation instead of distance",
        goal="Random partitions and path length: the idea that changed the field.",
        topics=("Isolation Forest", "Extended IForest"),
    ),
    StudyStep(
        stage="8",
        title="Event streams to daily features",
        goal="Aggregate raw sensor events into interpretable per-day features.",
        topics=("Window aggregation features",),
    ),
    StudyStep(
        stage="9",
        title="Point-process rhythm: IEI, CV and Fano factor",
        goal="Tell regular from bursty activity using the times between events.",
        topics=("Inter-event interval (IEI), CV and Fano factor",),
    ),
    StudyStep(
        stage="10",
        title="Sequence predictability: n-grams and entropy",
        goal="Quantify how predictable a sensor sequence is with Markov transitions.",
        topics=("N-gram / Markov transition entropy",),
    ),
    StudyStep(
        stage="11",
        title="Hidden regimes: the HMM",
        goal="Model latent states and score regime transitions across days.",
        topics=("HMM",),
    ),
    StudyStep(
        stage="12",
        title="Self-exciting events: the Hawkes process",
        goal="Model bursts: past events raising the rate of future ones.",
        topics=("Hawkes",),
    ),
    StudyStep(
        stage="13",
        title="Ensembles: combine, don't pick",
        goal="Understand why combining normalized detector scores beats trusting a single method.",
        topics=("Ensemble detectors",),
    ),
    StudyStep(
        stage="14",
        title="Evaluation without labels",
        goal="Read ROC / AUROC and understand synthetic anomaly injection as a proxy.",
        topics=("AUROC", "Synthetic anomaly injection"),
    ),
    StudyStep(
        stage="15",
        title="Applied: CASAS smart homes",
        goal="Put every piece together on real smart-home event streams.",
        topics=("CASAS", "Window aggregation features"),
    ),
)
