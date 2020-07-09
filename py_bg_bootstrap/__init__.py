# -*- coding: utf-8 -*-

"""Top-level package for py_bg_bootstrap."""

__author__ = "Jamie Sherman"
__email__ = "jamies@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.1"


def get_module_version():
    return __version__


from .bootstrapper import Bootstrapper  # noqa: F401
