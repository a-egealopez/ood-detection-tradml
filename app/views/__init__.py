"""View entry points exposed to the main app."""

from views.casas_view import render_casas_view
from views.data_view import DATA_2D, DATA_CASAS, render_data_step
from views.extractor_inspector import render_extractor_inspector
from views.playground_view import render_playground_view

__all__ = [
    "DATA_2D",
    "DATA_CASAS",
    "render_casas_view",
    "render_data_step",
    "render_extractor_inspector",
    "render_playground_view",
]
