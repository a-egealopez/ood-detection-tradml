"""Curated formal references: papers, surveys, library manuals and monographs.

Single source for the "Learn more" links shown on the detector cards (2D Playground)
and the feature-extraction methods (Features step). Every resource is a primary
paper, a peer-reviewed survey, an official library manual or a monograph - no blog
posts or wiki pages. Every URL was checked before being added.
"""

from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Resource model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Resource:
    """One curated reference for a detector or a feature-extraction method.

    ``method`` is the key that links the resource back to a detector name (the
    ``DETECTOR_REGISTRY`` display names) or to a feature-method key (the Features
    step). ``kind`` selects the badge shown in the UI.
    """

    method: str
    title: str
    kind: str  # paper | survey | manual | tool
    url: str
    note: str  # short didactic "why read this"


# Badge shown next to each resource kind.
KIND_LABELS = {
    "paper": ":material/description: Paper",
    "survey": ":material/menu_book: Survey",
    "manual": ":material/import_contacts: Docs",
    "tool": ":material/construction: Library",
}

# UI names that map onto a resource key (same references, two entry points).
ALIASES = {
    "Hidden Markov Model (HMM)": "HMM",
    "Hawkes self-exciting process": "Hawkes",
    "OC-SVM (RBF)": "OC-SVM",
    "OC-SVM (Linear)": "OC-SVM",
    "OC-SVM (Poly)": "OC-SVM",
    "Window Aggregation": "Window aggregation features",
    "Inter-Event Interval (IEI)": "Inter-event interval (IEI), CV and Fano factor",
    "N-gram Transition (Markov)": "N-gram / Markov transition entropy",
    "Next-Event Prediction (Markov)": "Next-event prediction (Markov)",
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
        note="Rabiner (1989), Proc. IEEE 77(2): the canonical introduction to states, emissions and the forward/backward machinery — 20k+ citations. Use it for the HMM formalism; "
        "note this detector goes one step further and scores each day by its causal predictive "
        "log-likelihood (conditioning on the history seen so far) to flag deviations.",
    ),
    Resource(
        method="HMM",
        title="hmmlearn",
        kind="tool",
        url="https://hmmlearn.readthedocs.io/",
        note="The library used in the sequential track; its tutorial maps 1:1 onto this detector.",
    ),
    Resource(
        method="HMM",
        title="Unsupervised Log Anomaly Detection with Few Unique Tokens",
        kind="paper",
        url="https://arxiv.org/abs/2310.08951",
        note="Sulc, Eichler & Wilksen (2023): scores each observation by the causal predictive "
        "log-likelihood (likelihood of the history vs. including the new point) — exactly the "
        "recursion this detector runs in numpy across days.",
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
        title="Statistical Models for Earthquake Occurrences and Residual Analysis for Point Processes",
        kind="paper",
        url="https://doi.org/10.1080/01621459.1988.10478560",
        note="Ogata (1988), JASA 83(401): the forward recursion for the conditional-intensity log-likelihood, "
        "which is exactly what this detector implements in numpy (no tick needed).",
    ),
    Resource(
        method="Hawkes",
        title="tick",
        kind="tool",
        url="https://x-datainitiative.github.io/tick/",
        note="The reference point-process library; the Hawkes detector here implements the "
        "exponential-kernel intensity recursion directly in numpy (Ogata 1988), so tick "
        "is not required.",
    ),
    # --- Feature-extraction methods (Features step) -------------------------
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
        method="Next-event prediction (Markov)",
        title="DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning",
        kind="paper",
        url="https://dl.acm.org/doi/10.1145/3133956.3134015",
        note="Du, Li, Zheng & Srikumar (2017), ACM CCS (~3000 citations): the seminal 'predict the next event and flag deviations' approach. Its paper explicitly frames the n-gram / Markov model as the classic baseline, which is exactly this extractor.",
    ),
    Resource(
        method="Next-event prediction (Markov)",
        title="Anomaly Detection for Discrete Sequences: A Survey",
        kind="paper",
        url="https://ieeexplore.ieee.org/document/6192365",
        note="Chandola, Mithal & Kumar (2012), IEEE TKDE 24(5): formal taxonomy of sequence anomalies; 'predictive models' are a canonical family — build a model of normal sequence behavior and flag deviations.",
    ),
    # --- Ensemble step -------------------------------------------------------
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
        title="On the Combination of Outlier Detection Algorithms",
        kind="paper",
        url="https://charuaggarwal.net/combine.pdf",
        note="Aggarwal (2013): average/maximization/AOM combination functions — the recipes behind the soft and hard voting here.",
    ),
    Resource(
        method="Ensemble detectors",
        title="PyOD model combination",
        kind="tool",
        url="https://pyod.readthedocs.io/",
        note="Practical score-combination recipes (average / maximization / AOM) used in real ensembles.",
    ),
)


def resources_for(method: str) -> tuple[Resource, ...]:
    """Curated references for a detector name or feature method (alias-resolved)."""
    key = ALIASES.get(method, method)
    return tuple(res for res in RESOURCES if res.method == key)
