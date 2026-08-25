import os
import sys
import traceback

import wx

_site_packages = os.path.normpath(os.path.join(os.path.dirname(__file__), "site-packages"))
if not any(os.path.normpath(p).lower() == _site_packages.lower() for p in sys.path):
    sys.path.insert(0, _site_packages)

try:
    from . import FreeDiffPair
except Exception:
    wx.LogMessage(traceback.format_exc())
