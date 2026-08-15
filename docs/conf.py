"""Sphinx configuration for the stonks / stonker docs."""

project = "stonks"
author = "Luis Chumpitaz Diaz"
copyright = "2026, Luis Chumpitaz Diaz"
release = "0.1.0"

extensions = [
    "myst_parser",
]
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
myst_title_to_anchor = True

html_theme = "alabaster"
html_theme_options = {
    "description": "Simple, exploratory algorithms for stocks and investments.",
    "github_user": "luisdiaz1997",
    "github_repo": "stonks",
    "github_banner": True,
}
