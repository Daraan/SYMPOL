from __future__ import annotations

from typing import TYPE_CHECKING

from ray_utilities.setup.scheduled_tuner_setup import PBTSetup
from sympol.rllib_port.sympol_setup import SympolSetup

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.ppo import PPO

    from sympol.rllib_port.core.algorithms import SympolPPOConfig
    from sympol.rllib_port.extended_args import SympolArgumentParser


class SympolPBTSetup(PBTSetup["SympolArgumentParser", "SympolPPOConfig", "PPO"], SympolSetup):
    """PBT setup for Sympol, analogous to MLPPBTSetup but for Sympol."""
