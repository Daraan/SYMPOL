from __future__ import annotations

if __name__ == "__main__":
    import os

    import ray

    # Call ray init to avoid jax os.fork RuntimeWarnings
    # see: https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray.init
    # ray debugger will not work if dashboard is not included
    ray.init(include_dashboard=os.environ.get("RAY_DEBUG", "1") == "1")

# Import comet (via ray_utilities) before other libraries (torch, tf, ...) to allow monkey patching.
from ray_utilities import run_tune  # fmt: skip
from rllib_port.sympol_setup import SympolSetup

if __name__ == "__main__":
    setup = SympolSetup(init_param_space=True)
    results = run_tune(setup)
