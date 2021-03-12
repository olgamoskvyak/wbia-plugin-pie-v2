# -*- coding: utf-8 -*-
from wbia_pie_v2 import _plugin  # NOQA

try:
    from wbia_pie_v2._version import __version__  # NOQA
except ImportError:
    __version__ = '0.0.0'
