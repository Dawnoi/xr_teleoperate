import sys

from inference import transport as _module

sys.modules[__name__] = _module
