from __future__ import annotations

from typing import TYPE_CHECKING

# Import comet before torch to allow monkey patching.
from ray_utilities import default_trainable, run_tune  # fmt: skip

from rllib_port.sympol.sympol_setup import SympolSetup

if TYPE_CHECKING:
    from ray_utilities.typing import FunctionalTrainable


def test_mode_func(trainable: FunctionalTrainable, setup: SympolSetup):
    return trainable({})
    return trainable(setup.param_space)


if __name__ == "__main__":
    setup = SympolSetup(init_param_space=True)
    results = run_tune(setup, test_mode_func)
