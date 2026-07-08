import sys

from inference import online_session as _module

sys.modules[__name__] = _module
