"""
PlacementPredict Central Configuration Bridge
=============================================
Proxies to 05_Web_Application/config.py so all pipeline scripts
in Src and 04_Models_and_Pipelines can seamlessly `import config`.
"""
import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.join(_curr_dir, "05_Web_Application")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from config import *
