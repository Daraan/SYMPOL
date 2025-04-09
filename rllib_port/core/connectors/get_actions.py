from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from ray.rllib.connectors.module_to_env import GetActions as _GetActions  # do not confuse with
from ray.rllib.core.columns import Columns

if TYPE_CHECKING:
    import chex

    from ray_utilities.jax.distributions.jax_distributions import RLlibToJaxDistribution
    from rllib_port.core.sympol_module import SympolPPOModule

__all__ = ["GetActionsJaxDistr"]


class GetActionsJaxDistr(_GetActions):
    # TODO: Set key during build /__init__

    def __init__(self, *args, key: chex.PRNGKey, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key

    def split_key(self):
        self.key, new_key = jax.random.split(self.key, 2)
        return new_key

    def _get_actions(self, batch: dict, sa_rl_module: SympolPPOModule, explore: bool):
        # Action have already been sampled -> Early out.
        if Columns.ACTIONS in batch:
            return

        # ACTION_DIST_INPUTS field returned by `forward_exploration|inference()` ->
        # Create a new action distribution object.
        if Columns.ACTION_DIST_INPUTS in batch:
            if explore:
                action_dist_class = sa_rl_module.get_exploration_action_dist_cls()
            else:
                action_dist_class = sa_rl_module.get_inference_action_dist_cls()
            action_dist = action_dist_class.from_logits(
                batch[Columns.ACTION_DIST_INPUTS],
            )
            if not explore:
                action_dist = action_dist.to_deterministic()

            # NEW:
            actions = action_dist.sample(seed=self.split_key())
            batch[Columns.ACTIONS] = actions

            # For convenience and if possible, compute action logp from distribution
            # and add to output.
            if Columns.ACTION_LOGP not in batch:
                batch[Columns.ACTION_LOGP] = action_dist.logp(actions)
