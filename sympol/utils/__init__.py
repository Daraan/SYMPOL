from .utils import *
from .utils import _is_discreteT as _is_discreteT
from .envs import *

# Will import ray
from ray_utilities.nice_logger import nice_logger

_logger = nice_logger(__name__, level="DEBUG")
