from dataclasses import dataclass, fields
from typing import Optional, TYPE_CHECKING, Literal
from typing_extensions import TypeGuard

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class _NoAttribute:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        raise AttributeError(
            f"Do use the no- attribute {self.name} directly use the corresponding attribute without the no_ prefix"
        )


class _HandleNos:
    def __post_init__(self: "DataclassInstance"):
        """Convert no_ arguments to their inverse without prefix"""
        for field_ in fields(self):
            key = field_.name
            value = getattr(self, key)
            if key.startswith("no_") and value is not None:
                setattr(self, key[3:], not value)
                setattr(self, key, _NoAttribute(key))


@dataclass(kw_only=True)
class GeneralArgs(_HandleNos):
    """General arguments for the experiment."""

    exp_name: str = "SYMPOL RL"
    """Experiment name, important for wandb tracking"""
    run_name: str = "Default"
    """Name for the current run"""
    device: str = "cuda"
    """Device to use for training"""
    track: bool = False
    """If true, initialize a wandb run and track the results"""
    seed: int = 42
    """Random seed"""
    env_id: str = "CartPole-v1"
    """Environment ID"""
    total_steps: int = 1_000_000
    """Number of total environment steps for training. If more than one environment is used, e.g. 5 environments, we have 5 total steps per env.step() call"""
    eval_freq: int = 50_000
    """Frequency of evaluation (in total timesteps)"""
    minibatch_size: int = 128
    """Minibatch size used for backpropagation/optimization"""
    n_eval_episodes: int = 5
    """Number of episodes for evaluation. The return of the evaluation is the mean of the cumulative reward across all evaluation episodes"""
    n_envs: int = 8
    """Number of environments to use for collecting data"""
    random_trials: int = 5
    """Number of random trials for evaluation"""


@dataclass(kw_only=True)
class Args(_HandleNos):
    """General arguments for the SYMPOL RL experiment."""

    use_best_config: bool = False
    """If true, use the already optimized config from configs.py. This might not exist yet for every environment, in this case the default values are used"""
    checkpoint: bool = False
    """If true, render environments each time there is an evaluation"""
    overwrite_explicit: bool = False
    """Whether to overwrite explicit arguments"""
    no_adamW: bool = True
    """Do not use AdamW optimizer (explicitly sets to False)"""
    adamW: bool = False
    """Whether to use AdamW optimizer"""
    normEnv: bool = False
    """Whether to normalize the environment"""
    use_batch_norm: bool = False
    """Whether to use batch normalization"""
    SWA: bool = False
    """Whether to use Stochastic Weight Averaging"""
    dynamic_buffer: bool = False
    """Use dynamic trajectory buffer"""
    static_batch: bool = False
    """Use static batch size"""
    path: str = "./checkpoints"
    """Path to save checkpoints"""
    render_each_eval: bool = False
    """If true, render environments each time there is an evaluation"""
    no_render_env: bool = False
    """Flag to disable rendering of the environment"""
    render_env: bool = True
    """Whether to render the environment"""
    no_reduce_lr: bool = False
    """Flag to not use reduce_lr"""
    reduce_lr: bool = True
    """Whether to use reduce_lr"""
    gpu_number: int = 0
    """GPU Number"""
    optimize_config: bool = False
    """If true, use optuna to optimize the parameters specified in the function body of suggest_config in configs.py"""
    n_trials: int = 100
    """Number of trials per optuna optimization job"""
    view_size: int = 3
    """MiniGrid view size"""


@dataclass(kw_only=True)
class PPOArgs(_HandleNos):
    """Arguments specific to the PPO algorithm."""

    actor: Literal["sympol", "mlp", "sdt", "d-sdt", "stateActionDT"] = "sympol"
    """Specify the actor type: 'sympol' or 'mlp' or 'sdt' 'stateActionDT'"""
    critic: Literal["mlp", "sdt", "sympol"] = "mlp"
    """Specify the critic type: 'mlp', 'sdt', or 'sympol'"""
    gamma: float = 0.99
    """Discount factor for future rewards"""
    gae_lambda: float = 0.95
    """General Advantage Estimator lambda parameter"""
    ent_coef: float = 0.01
    """Entropy coefficient, higher value corresponds to more exploration"""
    learning_rate_critic: float = 1e-3
    """Learning rate for the critic"""
    n_update_epochs: int = 5
    """Number of updates/gradient steps for each batch"""
    n_steps: int = 256
    """Number of steps to take in ONE environment before updating the policy / q-approximation parameters."""
    clip_vloss: bool = False
    """Clip value function loss"""
    clip_coef: float = 0.1
    """Clip coefficient PPO"""
    vf_coef: float = 0.5
    """Value function loss coefficient"""
    accumulate_gradients_every: int = 1
    """
    Number of accumulation steps for the gradient update.
    The accumulated gradients will be averaged before backpropagation
    """
    max_grad_norm: float = 0.5
    """Gradient clipping threshold"""
    target_kl: Optional[float] = None
    """Target KL divergence threshold"""
    norm_adv: bool = False
    """If true, Normalize the advantages"""


@dataclass(kw_only=True)
class ActorLearningRateSimple(_HandleNos):
    """Common arguments for SYMPOL, MLP, and SDT algorithms."""

    learning_rate_actor: float = 1e-3


@dataclass(kw_only=True)
class MLPModelArgs(ActorLearningRateSimple):
    """Arguments specific to the MLP algorithm."""

    num_layers: int = 2
    """Number of MLP layers"""
    neurons_per_layer: int = 256
    """Number of neurons per MLP layer"""


@dataclass(kw_only=True)
class SDTModelArgs(ActorLearningRateSimple):
    """Arguments specific to the SDT algorithm."""

    depth: int = 7
    """Depth for each single estimator/tree"""
    temperature: float = 1.0
    "SDT entmax temperature"

    action_type: Literal["discrete", "continuous"] = "discrete"
    """
    Type of the action space, i.e. discrete or continuous (classification vs regression).
    Continuous actions are not properly implemented yet though and will raise an NotImplementedException.
    """


@dataclass(kw_only=True)
class SYMPOLModelArgs(_HandleNos):
    """Arguments specific to the SYMPOL algorithm."""

    depth: int = 7
    """Depth for each single estimator/tree"""
    n_estimators: int = 1
    """Number of estimators/trees for the ensemble"""
    action_type: Literal["discrete", "continuous"] = "discrete"
    """
    Type of the action space, i.e. discrete or continuous (classification vs regression).
    Continuous actions are not properly implemented yet though and will raise an NotImplementedException.
    """
    learning_rate_actor_weights: float = 1e-3
    """Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)"""
    learning_rate_actor_split_values: float = 1e-3
    """Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)"""
    learning_rate_actor_split_idx_array: float = 1e-3
    """Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)"""
    learning_rate_actor_leaf_array: float = 1e-3
    """Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)"""
    learning_rate_actor_log_std: float = 1e-3
    """Learning rate for all weights in SYMPOL (estimator weights, split values, split indices and leaf classes)"""


# Combine Arguments


@dataclass
class SDTArgs(SDTModelArgs, ActorLearningRateSimple, PPOArgs, Args, GeneralArgs):
    """Arguments for the command line interface"""


@dataclass(kw_only=True)
class MLPArgs(MLPModelArgs, ActorLearningRateSimple, PPOArgs, Args, GeneralArgs):
    """Arguments for the command line interface"""


@dataclass(kw_only=True)
class SYMPOLArgs(SYMPOLModelArgs, PPOArgs, Args, GeneralArgs):
    """Arguments for the command line interface"""


@dataclass(kw_only=True)
class CLIArgs(SYMPOLArgs, MLPArgs, SDTArgs):
    """Arguments for the command line interface"""


def is_sympol_args(args: CLIArgs) -> TypeGuard[SYMPOLArgs]:
    return args.actor == "sympol"


def is_mlp_args(args: CLIArgs) -> TypeGuard[MLPArgs]:
    return args.actor == "mlp"


def is_a_sdt_args(args: CLIArgs) -> TypeGuard[SDTArgs]:
    return args.actor in ("sdt", "d-sdt")
