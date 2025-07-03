from ray_utilities.nice_logger import nice_logger

logger = nice_logger(__name__, level="DEBUG")

try:
    from . import rllib_port
    from . import args
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    import rllib_port
    import args
