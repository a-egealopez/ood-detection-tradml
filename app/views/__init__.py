"""View entry points exposed to the main app."""

from views.casas_view import render_casas_view
from views.documentation_view import render_documentation_view
from views.feature_extraction_view import render_feature_extraction_view
from views.playground_view import render_playground_view

__all__ = [
    "render_casas_view",
    "render_documentation_view",
    "render_feature_extraction_view",
    "render_playground_view",
]
