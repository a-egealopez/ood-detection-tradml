"""
Visualizador interactivo de detectores de anomalías en datasets sintéticos.
Renderiza en tiempo real con Plotly cuando cambian los parámetros.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Any


def create_anomaly_scatter(
    X: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray = None,
    detector_name: str = "Detector",
    title: str = "Anomaly Detection",
) -> go.Figure:
    """
    Crea scatter plot interactivo de resultados de detección.
    
    Args:
        X: (n_samples, 2) datos 2D
        y_pred: (n_samples,) predicciones binarias (0=normal, 1=anomalía)
        scores: (n_samples,) scores continuos [0, 1]
        y_true: (n_samples,) etiquetas verdaderas (opcional)
        detector_name: nombre del detector
        title: título del gráfico
    
    Returns:
        Figura Plotly interactiva
    """
    
    colors = np.where(y_pred == 1, "crimson", "steelblue")
    
    fig = go.Figure()
    
    # Puntos normales
    normal_mask = y_pred == 0
    fig.add_trace(go.Scatter(
        x=X[normal_mask, 0],
        y=X[normal_mask, 1],
        mode="markers",
        marker=dict(
            size=6,
            color="steelblue",
            opacity=0.6,
            line=dict(width=0),
        ),
        name="Normal",
        hovertemplate="<b>Normal</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
    ))
    
    # Puntos anómalos (predichos)
    anomaly_mask = y_pred == 1
    fig.add_trace(go.Scatter(
        x=X[anomaly_mask, 0],
        y=X[anomaly_mask, 1],
        mode="markers",
        marker=dict(
            size=8,
            color=scores[anomaly_mask],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Anomaly<br>Score", thickness=10, len=0.6),
            line=dict(width=1, color="darkred"),
        ),
        name="Predicted Anomaly",
        hovertemplate="<b>Anomaly</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Score: %{marker.color:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{detector_name}</b><br><sub>{title}</sub>",
            font=dict(size=14),
        ),
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        hovermode="closest",
        height=500,
        template="plotly_white",
        showlegend=True,
    )
    
    return fig


def create_score_distribution(
    scores: np.ndarray,
    y_pred: np.ndarray,
    detector_name: str = "Detector",
) -> go.Figure:
    """Histograma de distribución de scores con separación por clase."""
    
    fig = go.Figure()
    
    normal_scores = scores[y_pred == 0]
    anomaly_scores = scores[y_pred == 1]
    
    fig.add_trace(go.Histogram(
        x=normal_scores,
        nbinsx=25,
        name="Normal",
        marker_color="steelblue",
        opacity=0.7,
    ))
    
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        nbinsx=25,
        name="Anomaly",
        marker_color="crimson",
        opacity=0.7,
    ))
    
    fig.update_layout(
        title=f"<b>{detector_name}</b> - Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )
    
    return fig


def create_comparison_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
) -> go.Figure:
    """
    Crea tabla/gráfico comparativo de métricas entre detectores.
    
    Args:
        metrics_dict: {detector_name: {metric_name: value}}
    """
    
    detector_names = list(metrics_dict.keys())
    
    metrics = {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "AUROC": [],
    }
    
    for det_name in detector_names:
        for metric_key in metrics.keys():
            if metric_key in metrics_dict[det_name]:
                metrics[metric_key].append(metrics_dict[det_name][metric_key])
            else:
                metrics[metric_key].append(0.0)
    
    fig = go.Figure()
    
    for metric_name, values in metrics.items():
        fig.add_trace(go.Bar(
            name=metric_name,
            x=detector_names,
            y=values,
            hovertemplate="<b>%{x}</b><br>" + metric_name + ": %{y:.3f}<extra></extra>",
        ))
    
    fig.update_layout(
        title="<b>Detector Performance Comparison</b>",
        xaxis_title="Detector",
        yaxis_title="Score",
        barmode="group",
        height=400,
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(range=[0, 1.05]),
    )
    
    return fig


def create_parameter_grid_analysis(
    param_name: str,
    param_values: np.ndarray,
    metrics_per_param: Dict[str, list],
    detector_name: str = "Detector",
) -> go.Figure:
    """
    Visualiza cómo varía una métrica al cambiar un parámetro.
    Útil para entender sensibilidad del detector.
    """
    
    fig = go.Figure()
    
    for metric_name, values in metrics_per_param.items():
        fig.add_trace(go.Scatter(
            x=param_values,
            y=values,
            mode="lines+markers",
            name=metric_name,
            hovertemplate=f"{param_name}: %{{x:.3f}}<br>" + 
                         metric_name + ": %{y:.3f}<extra></extra>",
        ))
    
    fig.update_layout(
        title=f"<b>{detector_name}</b> - Sensitivity to {param_name}",
        xaxis_title=param_name,
        yaxis_title="Metric Score",
        height=400,
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(range=[0, 1.05]),
    )
    
    return fig


if __name__ == "__main__":
    # Test: generar datos aleatorios y visualizar
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y_pred = np.random.binomial(1, 0.1, size=100)
    scores = np.random.uniform(0, 1, size=100)
    
    fig1 = create_anomaly_scatter(X, y_pred, scores, detector_name="Test Detector")
    fig1.show()
    
    fig2 = create_score_distribution(scores, y_pred, detector_name="Test Detector")
    fig2.show()