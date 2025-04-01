from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2

# from ray.rllib.core.columns import Columns
if TYPE_CHECKING:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule


class RemoveStateFromBatch(ConnectorV2):
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        batch: dict[str, Any],
        shared_data: Optional[dict] = None,
        **kwargs,
    ):
        for module_id, module in rl_module.items():
            # remove actor or critic state
            # TODO: Make no throwing exception
            batch[module_id].pop("state")
        return batch
