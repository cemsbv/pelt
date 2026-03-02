import pelt
import numpy

project = 'pelt'
copyright = '2026, Thomas Versteeg'
author = 'Thomas Versteeg'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
