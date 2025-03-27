import logging
from rllib_port.model_interface import FlaxRLModel
from sympol import SYMPOL_RL

logger = logging.getLogger(__name__)


@struct.dataclass(kw_only=True, frozen=False)
class SympolRLModel(SYMPOL_RL, FlaxRLModel):


if TYPE_CHECKING:
    SympolRLModel(obs_dim=0, action_dim=0, action_type="discrete", depth=1, n_estimators=1)
