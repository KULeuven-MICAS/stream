# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
project = "Stream"
copyright = f"2023–{datetime.now().year}, MICAS, KU Leuven"

author = "Arne Symons"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    # Uncomment these if needed:
    # "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    # "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# Furo-compatible theme options
html_theme_options = {
    "light_logo": "stream_horizontal_logo_light.svg",
    "dark_logo": "stream_horizontal_logo_dark.svg",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/kuleuven-micas/stream/",
    "source_branch": "master",
    "source_directory": "docs/source/",
    # Color customization
    "light_css_variables": {
        # Brand color (used for links, TOC highlights, etc.)
        "color-brand-primary": "#2B6CB0",  # A calm, rich blue
        "color-brand-content": "#1A202C",  # Near-black for main content text
        "color-admonition-title-background": "#EBF8FF",  # Pale blue for note titles
        "color-admonition-background": "#FCF7F7",  # Very light grey-blue background
    },
    "dark_css_variables": {
        "color-brand-primary": "#63B3ED",  # Lighter blue for contrast
        "color-brand-content": "#E2E8F0",  # Soft light grey for readability
        "color-admonition-title-background": "#2C5282",  # Muted blue block titles
        "color-admonition-background": "#1A202C",  # Deep grey-blue for content
    },
}

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
    ]
}

html_static_path = ["_static"]
