from typing import Optional, Protocol
from typing_extensions import NotRequired, Literal, TypeGuard, TypeAliasType, TypedDict


class GeneralArgsDict(TypedDict, total=True):
    exp_name: str
    run_name: str
    device: str
    track: bool
    seed: int
    env_id: str
    total_steps: int
    eval_freq: int
    minibatch_size: int
    n_eval_episodes: int
    n_envs: int
    random_trials: int


class ArgsDict(TypedDict, total=True):
    use_best_config: bool
    checkpoint: bool
    overwrite_explicit: bool
    adamW: bool
    normEnv: bool
    use_batch_norm: bool
    SWA: bool
    dynamic_buffer: bool
    static_batch: bool
    path: str
    render_each_eval: bool
    render_env: bool
    reduce_lr: bool
    gpu_number: int
    optimize_config: bool
    n_trials: int
    view_size: int


class PPOArgsDict(TypedDict, total=True):
    actor: Literal["sympol", "mlp", "sdt", "d-sdt", "stateActionDT"]
    critic: Literal["mlp", "sdt", "sympol"]
    gamma: float
    gae_lambda: float
    ent_coef: float
    learning_rate_critic: float
    n_update_epochs: int
    n_steps: int
    clip_vloss: bool
    clip_coef: float
    vf_coef: float
    accumulate_gradients_every: int
    max_grad_norm: float
    target_kl: Optional[float]
    norm_adv: bool


class MLPModelArgsDict(TypedDict, total=True):
    num_layers: int
    neurons_per_layer: int

    learning_rate_actor: float


class SDTModelArgsDict(TypedDict, total=True):
    depth: int
    temperature: float
    action_type: Literal["discrete", "continuous"]

    learning_rate_actor: float


class SYMPOLModelArgsDict(TypedDict, total=True):
    depth: int
    n_estimators: int
    action_type: Literal["discrete", "continuous"]
    learning_rate_actor_weights: float
    learning_rate_actor_split_values: float
    learning_rate_actor_split_idx_array: float
    learning_rate_actor_leaf_array: float
    learning_rate_actor_log_std: float

    adamW: bool

    dropout: NotRequired[float]
    """Unused"""


class SDTArgsDict(SDTModelArgsDict, PPOArgsDict, ArgsDict, GeneralArgsDict):
    pass


class MLPArgsDict(MLPModelArgsDict, PPOArgsDict, ArgsDict, GeneralArgsDict):
    pass


class SYMPOLArgsDict(SYMPOLModelArgsDict, PPOArgsDict, ArgsDict, GeneralArgsDict):
    pass


class CLIArgsDict(SYMPOLArgsDict, MLPArgsDict, SDTArgsDict):
    pass


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
    dropout: NotRequired[float]
    """Unused"""
    n_estimators: int
    """Always 1"""


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
