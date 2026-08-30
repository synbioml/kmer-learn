"""Sphinx configuration for the kmer-learn documentation."""

from importlib.metadata import version

project = "kmer-learn"
author = "Arsenii Zinkevich"
release = version("kmer-learn")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3
nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
nb_merge_streams = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

html_theme = "furo"
html_title = f"kmer-learn {version}"
html_theme_options = {
    "source_repository": "https://github.com/synbioml/kmer-learn/",
    "source_branch": "master",
    "source_directory": "docs/",
}
