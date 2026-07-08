import sys

from inference import pose_transform as _module

sys.modules[__name__] = _module
