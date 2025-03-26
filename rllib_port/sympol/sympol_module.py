from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.models.catalog import ModelCatalog  # deprecated
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT

# see https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/modelv2_to_rlm.py
# for a Intermediate old API to new API Module

class SympolPPOModule(DefaultPPORLModule):
    # torch/tf specific
    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        """Computes the value estimates given `batch`.

        Args:
            batch: The batch to compute value function estimates for.
            embeddings: Optional embeddings already computed from the `batch` (by
                another forward pass through the model's encoder (or other subcomponent
                that computes an embedding). For example, the caller of thie method
                should provide `embeddings` - if available - to avoid duplicate passes
                through a shared encoder.

        Returns:
            A tensor of shape (B,) or (B, T) (in case the input `batch` has a
            time dimension. Note that the last value dimension should already be
            squeezed out (not 1!).
        """
        if embeddings is None:
            # Separate vf-encoder.
            if hasattr(self.encoder, "critic_encoder"):
                batch_ = batch
                if self.is_stateful():
                    # The recurrent encoders expect a `(state_in, h)`  key in the
                    # input dict while the key returned is `(state_in, critic, h)`.
                    batch_ = batch.copy()
                    batch_[Columns.STATE_IN] = batch[Columns.STATE_IN][CRITIC]
                embeddings = self.encoder.critic_encoder(batch_)[ENCODER_OUT]
            # Shared encoder.
            else:
                embeddings = self.encoder(batch)[ENCODER_OUT][CRITIC]

        # Value head.
        vf_out = self.vf(embeddings)
        # Squeeze out last dimension (single node value head).
        return vf_out.squeeze(-1)  # XXX


if TYPE_CHECKING:
    SympolPPOModule()
