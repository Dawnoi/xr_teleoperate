import sys

from inference import rtc_session as _module

sys.modules[__name__] = _module
