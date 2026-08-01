"""Documentation view: project overview, architecture and methodology."""

import streamlit as st


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

            Two learning tracks are exposed:
            - **Teaching track**: inspect each detector's real decision boundary on
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
