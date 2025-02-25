# -*- coding: utf-8 -*-
#
# data-science-template documentation build configuration file, created by
# sphinx-quickstart.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(
    0, os.path.join(os.path.abspath(".."), os.path.abspath("py_fatigue"))
)
from py_fatigue.version import __version__, parse_version

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "autoclasstoc",
    "sphinx_design",
    "sphinx_copybutton",
]

# enable autosummary
autosummary_generate = False
autosummary_imported_members = False
autodoc_member_order = "bysource"
autoclass_content = "both"

myst_enable_extensions = [
    "colon_fence",
]
suppress_warnings = ["myst.header"]

intersphinx_mapping = {"python": ("https://docs.python.org/3/", None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "py-fatigue"
author = "Pietro D'Antuono"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
major, minor, patch, _, _ = parse_version(__version__)
version = f"{major}.{minor}.{patch}"
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates", "_autosummary"]
# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "colorful"

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ["py_fatigue"]


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme =  "sphinx_rtd_theme"  # "cloud"
html_theme = "pydata_sphinx_theme"
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
fontcssurl = (
    "https://fonts.googleapis.com/css?family="
    + "Roboto|Alegreya|Ubuntu%20Mono|Merriweather|EB%20Garamond"
)

# Define the json_url for our version switcher.
json_url = "https://pydata-sphinx-theme.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.
# version_match = os.environ.get("READTHEDOCS_VERSION")

html_theme_options = {
    "navbar_align": "right",
    "external_links": [
        {
            "url": "https://github.com/OWI-Lab/py_fatigue/blob/main/CHANGELOG.md",
            "name": "Changelog",
        },
        {
            "url": "https://pydata.org",
            "name": "PyData",
        },
    ],
    "github_url": "https://github.com/owi-lab/py_fatigue",
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/py-fatigue/",
            "icon": "fa-brands fa-python fa-lg",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/owi-lab",
            "icon": "fa-brands fa-linkedin-in  fa-lg",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "OWI-Lab",
            "url": "https://www.owi-lab.be",
            "icon": "_static/_img/owi-lab.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "show_toc_level": 4,
    # "navbar_center": ["version-switcher", "navbar-nav"],
    # "announcement": "https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/main/docs/_templates/custom-template.html",
    # "show_nav_level": 2,
    # "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "footer_items": ["copyright", "sphinx-version"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    # "switcher": {
    #     "json_url": json_url,
    #     "version_match": version_match,
    # },
    # "search_bar_position": "navbar",  # TODO: Deprecated - remove in future version
}

# html_theme_options = {
#     "analytics_id": "G-XXXXXXXXXX",  #  Provided by Google in your dashboard
#     "analytics_anonymize_ip": False,
#     "logo_only": False,
#     "display_version": True,
#     "prev_next_buttons_location": "bottom",
#     "style_external_links": True,
#     "vcs_pageview_mode": "",
#     "style_nav_header_background": "#55A5D9",
#     # Toc options
#     "collapse_navigation": True,
#     "sticky_navigation": True,
#     "navigation_depth": 4,
#     "includehidden": True,
#     "titles_only": False,
# }
# html_theme_options = {
#     # 'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
#     "max_width": "10.5in",
#     "headfont": "Merriweather",
#     "bodyfont": "EB Garamond",
#     "codeblockfont": "Ubuntu Mono",
#     "fontcssurl": fontcssurl,
#     "footerbgcolor": "#4A567E",
#     "relbarbgcolor": "#4675B8",
#     "sectionbgcolor": "#0E76BC",
#     "rubricbgcolor": "#59b5f3",
#     "object_class_color": "#FFE0B2",
#     "object_attribute_color": "#B2EBF3",
#     "object_function_color": "#C8E6C9",
#     "object_exception_color": "#F7CFD3",
#     # "footerbgcolor": "435580",
#     "codebgcolor": "#FFF9E4",  # "#ffe0cc",
#     "codetrimcolor": "#F8D186",
#     "highlightcolor": "#F8D186",
#     # "bgcolor": "",
#     # "bgcolor": "",
#     # "bgcolor": "",
#     # "bgcolor": "",
#     # 'default_layout_text_size': '11px',
#     "lighter_header_decor": True,
#     # Toc options
#     # "collapse_navigation": False,
#     # "sticky_navigation": True,
#     # "navigation_depth": 6,
#     # "includehidden": True,
#     # "titles_only": False,
# }
html_css_files = [
    "css/custom.css",
]
# html_js_files = [
#     'js/custom.js',
# ]
mathjax3_config = {
    "chtml": {
        "mtextInheritFont": "true",
    },
}
# mathjax2_config = {
#     'extensions': ['tex2jax.js'],
#     'jax': ['input/TeX', 'output/HTML-CSS'],
#     'preferredFonts': "TeX",
#     'webFont':"",
#     'imageFont':"",
#     'undefinedFamily':"'Neo Euler', serif"
# }
# mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
# html_style = 'css/custom.css'
# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"py-fatigue Documentation ({version})"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = os.path.join(html_static_path[0], "_img", "py-fatigue-logo.png")

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = os.path.join(html_static_path[0], "_img", "owi-lab.png")

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "py_fatigue-templatedoc"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "py_fatigue.tex",
        "py_fatigue Documentation",
        "kapernikov",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        "index",
        "data-science-template",
        "data-science-template Documentation",
        ["kapernikov"],
        1,
    )
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "data-science-template",
        "data-science-template Documentation",
        "kapernikov",
        "data-science-template",
        "Testing template",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'
