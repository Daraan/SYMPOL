from __future__ import annotations

from typing import TYPE_CHECKING

if __name__ == "__main__":
    import ray
    import sys

    # Call ray init to avoid jax os.fork RuntimeWarnings
    # see: https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray.init
    ray.init(include_dashboard=len(sys.argv) > 1 and sys.argv[1] != "--test")

# Import comet before torch to allow monkey patching.
from ray_utilities import run_tune  # fmt: skip
from rllib_port.sympol.sympol_setup import SympolSetup

if TYPE_CHECKING:
    from ray_utilities.typing import FunctionalTrainable


def test_mode_func(trainable: FunctionalTrainable, setup: SympolSetup):
    return trainable({})
    return trainable(setup.param_space)


if __name__ == "__main__":
    setup = SympolSetup(init_param_space=True)
    results = run_tune(setup, test_mode_func)
