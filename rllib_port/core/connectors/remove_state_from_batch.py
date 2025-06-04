# ruff: noqa: ARG002
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ray.rllib.connectors.connector_v2 import ConnectorV2
from typing_extensions import deprecated

# from ray.rllib.core.columns import Columns
if TYPE_CHECKING:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

__NO_STATE_KEY = object()


@deprecated("The state key is not longer added to the batch.")
class RemoveStateFromBatch(ConnectorV2):
    """
    Attentions:
        Must be used with a MultiRLModule. This is normally the case but not guaranteed by the
        interface.
    """

    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        batch: dict[str, Any],
        **kwargs,
    ):
        for module_id in rl_module.keys():
            # remove actor or critic state
            if batch[module_id].pop("state", __NO_STATE_KEY) is __NO_STATE_KEY:
                # If no state was present, we do not throw an error.
                continue
        return batch
