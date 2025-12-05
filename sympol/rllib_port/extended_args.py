from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Any, Dict, Literal

from typing_extensions import Self, deprecated

from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from sympol.args import ArgumentParserWithDefaults
from sympol.config_types.args_types import SympolCLIArgs

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)


# circular import define after class
def get_args():
    return SympolArgumentParser().parse_args()


def _float_or_none(value: str) -> float | None:
    if value.lower() in ("none", "null"):
        return None
    return float(value)


_NOT_FOUND = object()


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

    render_env: bool = False
    reduce_lr: bool = False

    if TYPE_CHECKING:  # cannot overwrite attribute

        @property
        @deprecated("Use dynamic_batch instead")
        def static_batch(self) -> Literal[False]:  # pyright: ignore
            """The value of static_batch is not reliable, as it is forwarded to dynamic_batch"""
            ...

    def parse_args(self, args=None, *, known_only=False, **kwargs) -> Self:
        # Ensure explicit args are tracked by calling parse_known_args first
        if args is None:
            args = sys.argv[1:]
        # Set explicit args manually before calling super
        self._explicit_args = {arg[2:] for arg in args if arg.startswith("--")}
        # Call the full chain to get proper dataclass processing
        return super(ArgumentParserWithDefaults, self).parse_args(args, known_only=known_only, **kwargs)  # type: ignore[return-type]

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
            "--render_env",
            action="store_true",
            help="Flag to enable rendering of the environment",
            required=False,
            default=False,
        )
        # self.add_argument(
        #    "--no-render_env",
        #    dest="render_env",
        #    action="store_false",
        #    help="Flag to disable rendering of the environment",
        #    required=False,
        # )
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

        # legacy args for grad_clip
        self.add_argument(
            "--max-grad-norm",
            type=_float_or_none,
            dest="grad_clip",
            default=DefaultArgumentParser.grad_clip,
            help="Maximum gradient norm for clipping",
        )

        # no args from fields

    # overwritten by dataclass from CLI Args

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d.pop("use_comet_offline", None)  # do not set property
        # TODO: There are more properties now
        return super().__setstate__(d)

    def process_args(self) -> None:
        super().process_args()
        if self.adamW is True:
            logger.error("AdamW is already True")
        if self.seed is None and type(self).seed is not None:  # pyright: ignore[reportUnnecessaryComparison]
            logger.error("No seed found. But there should be one in the class. This should not happen.")
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
        # Handle env_type <-> env_id mapping
        # env_type is from DefaultArgumentParser, env_id is from SympolCLIArgs
        if hasattr(self, "env_type") and hasattr(self, "env_id"):
            # If env_type was set explicitly (e.g., from command line), use it for env_id
            if "env_type" in getattr(self, "_explicit_args", set()):
                self.env_id = self.env_type
            # Always ensure env_type matches env_id for compatibility
            self.env_type = self.env_id
        elif hasattr(self, "env_id"):
            self.env_type = self.env_id

        # Handle agent_type <-> actor mapping
        # agent_type is from DefaultArgumentParser, actor is from SympolCLIArgs
        if hasattr(self, "agent_type") and hasattr(self, "actor"):
            # If agent_type was set explicitly (e.g., from command line), use it for actor
            if "agent_type" in getattr(self, "_explicit_args", set()):
                self.actor = self.agent_type
            # Always ensure agent_type matches actor for compatibility
            self.agent_type = self.actor
        elif hasattr(self, "actor"):
            self.agent_type = self.actor

        self.render_mode = "rgb_array" if self.render_env else None

    def _process_args_sympol(self) -> None:
        explicit_args_corrected = []
        for some_arg in self._explicit_args:
            if "no-" in some_arg:
                explicit_args_corrected.append("".join(some_arg.split("no-")))
            else:
                explicit_args_corrected.append(some_arg)
        # Map CLI argument names to their destination attributes
        dest_mapping = {}
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                continue
            if action.dest != action.option_strings[0].lstrip("-").replace("-", "_"):
                dest_mapping[action.option_strings[0].lstrip("-").replace("-", "_")] = action.dest

        # Handle argument mappings between DefaultArgumentParser and SympolCLIArgs
        # Map env_type -> env_id (DefaultArgumentParser uses env_type, SympolCLIArgs uses env_id)
        arg_mappings = {"env_type": "env_id", "agent_type": "actor"}

        explicit_arg_values = {}
        for arg in explicit_args_corrected:
            # Convert hyphens to underscores for attribute lookup
            arg_normalized = arg.replace("-", "_")
            dest_arg = dest_mapping.get(arg_normalized, arg_normalized)
            mapped_arg = arg_mappings.get(dest_arg, dest_arg)
            attr = getattr(self, dest_arg, _NOT_FOUND)
            if attr is not _NOT_FOUND:
                explicit_arg_values[mapped_arg] = attr

        no_update = True
        if self.use_best_config:
            import sympol.configs as configs

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
