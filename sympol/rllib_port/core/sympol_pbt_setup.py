from __future__ import annotations

import os
from typing import TYPE_CHECKING


from ray_utilities.setup.scheduled_tuner_setup import PBTTunerSetup
from sympol.rllib_port.sympol_setup import SympolSetup

if TYPE_CHECKING:
    from sympol.rllib_port.extended_args import SympolArgumentParser


class SympolPBTSetup(SympolSetup):
    """PBT setup for Sympol, analogous to MLPPBTSetup but for Sympol."""

    _tuner_setup_cls = PBTTunerSetup

    def _tuner_add_iteration_stopper(self):
        """PBT handles stopping."""
        return False

    def create_tuner(self, *, adv_loggers: bool | None = None):
        if self.args.command_str != "pbt":
            raise RuntimeError(f"{type(self)} requires 'pbt' command, got '{self.args.command_str}'")
        # Save trial state every 15 minutes as PBT can be long running, can take ~1 min to save
        if os.environ.get("RAY_UTILITIES_NO_PBT_CHECKPOINT_CHANGE") != "1":
            os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = str(60 * 15)
        assert self.args.command is not None
        # NOTE: Uses args.metrics/mode not the args.command.metric/mode
        return super().create_tuner(adv_loggers=True if adv_loggers is None else adv_loggers)
