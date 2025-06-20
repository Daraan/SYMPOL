from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal

from typing_extensions import Self, deprecated

from args import ArgumentParserWithDefaults
from config_types.args_types import SympolCLIArgs
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)


# circular import define after class
def get_args():
    return SympolArgumentParser().parse_args()


class SympolArgumentParser(ArgumentParserWithDefaults, DefaultArgumentParser, SympolCLIArgs):
    # Make Non-Required
    agent_type: Literal["sympol", "mlp", "sdt", "d-sdt", "stateActionDT"] = "sympol"
    """Sync with args.actor"""

    seed: int = 42  # pyright: ignore[reportIncompatibleVariableOverride]
    """
    Seed for the environment

    Note:
        For JAX `None` is not a valid value.
    """

    legacy: bool = False
    """Use original SYMPOL implementation for PPO and batching"""

    if TYPE_CHECKING:  # cannot overwrite attribute

        @property
        @deprecated("Use dynamic_batch instead")
        def static_batch(self) -> Literal[False]:  # pyright: ignore
            """The value of static_batch is not reliable, as it is forwarded to dynamic_batch"""
            ...

    def parse_args(self, args=None, *, known_only=False, **kwargs) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        # this will call ArgumentParserWithDefaults.parse_unknown_args which sets explicit args
        return DefaultArgumentParser.parse_args(self, args, known_only=known_only, **kwargs)  # type: ignore[return-type]

    def configure(self) -> None:
        super().configure()

        self.add_argument(
            "--no-adamW",
            action="store_false",
            dest="adamW",
            help="Do not use AdamW optimizer (explicitly sets to False)",
            required=False,
            default=False,  # will be stored in adamW
        )
        self.add_argument(
            "--no-render_env",
            dest="render_env",
            action="store_false",
            help="Flag to disable rendering of the environment",
            required=False,
        )
        self.add_argument(
            "--no-reduce_lr",
            dest="reduce_lr",
            action="store_false",
            help="Flag to not use reduce_lr",
            required=False,
        )
        # Overwrite to to use new default
        self.add_argument("-s", "--seed", default=42, type=int)
        # NOTE: DefaultArgumentParser uses dynamic_batch=False
        self.add_argument("--static_batch", action="store_false", dest="dynamic_batch", required=False, default=False)

        # no args from fields

    # overwritten by dataclass from CLI Args

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d.pop("use_comet_offline", None)  # do not set property
        return super().__setstate__(d)

    def process_args(self) -> None:
        super().process_args()
        if self.adamW is True:
            logger.error("AdamW is already True")
        if self.seed is None and type(self).seed is not None:
            logger.error("No seed found. But there should be one in the class. Should not happen.")
        self._process_args_sympol()
        self._process_args_ray_utilities()
        assert self.agent_type == self.actor
        if self.agent_type in ["mlp", "sdt", "d-sdt", "stateActionDT"]:
            if self.accumulate_gradients_every != 1:
                logger.warning(
                    "Accumulating gradients is not used for agent type: %s. Setting accumulate_gradients_every = 1 ",
                    self.agent_type,
                )
            self.accumulate_gradients_every = 1  # do not accumulate gradients

    def _process_args_ray_utilities(self) -> None:
        """Make CLIArgs compatible with DefaultArgumentParser"""
        # self.iterations = int(self.total_steps / self.n_envs / self.minibatch_size)
        self.env_type = self.env_id
        self.agent_type = self.actor  # pyright: ignore[reportIncompatibleVariableOverride]
        self.render_mode = "rgb_array" if self.render_env else None

    def _process_args_sympol(self) -> None:
        explicit_args_corrected = []
        for some_arg in self._explicit_args:
            if "no-" in some_arg:
                explicit_args_corrected.append("".join(some_arg.split("no-")))
            else:
                explicit_args_corrected.append(some_arg)
        # FIXME: When using args with "dest" these will not match the dest
        explicit_arg_values = {arg: getattr(self, arg) for arg in explicit_args_corrected}

        no_update = True
        if self.use_best_config:
            import configs

            for name, value in vars(configs).items():
                if "minigrid" in name and name.split("_")[1] in self.env_id.lower():
                    if self.actor == "stateActionDT":
                        best_cfg = value["mlp"]
                    elif self.actor == "d-sdt":
                        best_cfg = value["sdt"]
                    else:
                        best_cfg = value[self.actor]
                    logger.info("--- Updating best config %s", best_cfg)
                    self.__dict__.update(best_cfg)
                    no_update = False
                    break
                if name == "-".join(self.env_id.lower().split("-")[:-1]):
                    if self.actor == "stateActionDT":
                        best_cfg = value["mlp"]
                    elif self.actor == "d-sdt":
                        best_cfg = value["sdt"]
                    else:
                        best_cfg = value[self.actor]
                    logger.info("--- Updating best config %s", best_cfg)
                    self.__dict__.update(best_cfg)
                    no_update = False
                    break
        if self.overwrite_explicit:
            logger.info("--- Updating explicit args %s", explicit_arg_values)
            self.__dict__.update(explicit_arg_values)
            no_update = False
        else:
            # update at least own and super args
            logger.info("--- Updating explicit args %s not in CLIArgs", explicit_arg_values)
            self.__dict__.update(
                {k: v for k, v in explicit_arg_values.items() if k not in SympolCLIArgs.__dataclass_fields__}
            )

        if no_update:
            logger.debug(" --- No update of args ---")

    def __post_init__(self: DataclassInstance):
        # fix dataclass default values
        super().__post_init__()
        # Dataclass default comes from CLIArgs
        self.__dataclass_fields__["total_steps"].default = SympolArgumentParser.total_steps
