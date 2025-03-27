import logging

from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.deprecation import logger as __deprecation_logger

# This suppresses a deprecation warning from RLModuleConfig
__old_level = __deprecation_logger.getEffectiveLevel()
__deprecation_logger.setLevel(logging.ERROR)
RLModuleConfig()
__deprecation_logger.setLevel(__old_level)
