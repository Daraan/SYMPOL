from typing import Protocol, TypedDict
from typing_extensions import NotRequired, Literal, TypeGuard, TypeAliasType


class GeneralParams(TypedDict):
    learning_rate_critic: float
    reduce_lr: bool
    minibatch_size: int
    n_update_epochs: int
    max_grad_norm: float
    norm_adv: bool
    ent_coef: float
    vf_coef: float
    gamma: float
    gae_lambda: float
    n_steps: NotRequired[int]
    n_envs: NotRequired[int]


class MLPParams(GeneralParams):
    learning_rate_actor: float
    num_layers: int
    neurons_per_layer: int
    adamW: NotRequired[bool]


class SDTParams(GeneralParams):
    depth: int
    learning_rate_actor: float
    adamW: NotRequired[bool]
    critic: Literal["mlp", "sdt"]
    temperature: float


class SympolParams(GeneralParams):
    depth: int
    learning_rate_actor_weights: float
    learning_rate_actor_split_values: float
    learning_rate_actor_split_idx_array: float
    learning_rate_actor_leaf_array: float
    learning_rate_actor_log_std: float
    learning_rate_critic: float
    SWA: bool
    adamW: bool
    reduce_lr: bool
    dropout: float
    n_estimators: NotRequired[int]


class _HasActor(Protocol):
    actor: str


def is_sympol_params(params: GeneralParams, args: _HasActor) -> TypeGuard[SympolParams]:  # noqa: ARG001
    """Equivalent of args.actor == "sympol"""
    return args.actor == "sympol"


def is_mlp_params(params: GeneralParams, args: _HasActor) -> TypeGuard[MLPParams]:  # noqa: ARG001
    """Equivalent of args.actor in ("mlp", "stateActionDT")"""
    return args.actor in ("mlp", "stateActionDT")


def is_sdt_params(params: GeneralParams, args: _HasActor) -> TypeGuard[SDTParams]:  # noqa: ARG001
    """Equivalent of args.actor in ("sdt", "d-sdt")"""
    return args.actor in ("sdt", "d-sdt")


ParamsDictType = TypeAliasType("ParamsDictType", MLPParams | SDTParams | SympolParams)
