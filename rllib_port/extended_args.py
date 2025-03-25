import logging

from tap import Tap
from typing_extensions import Self

from args import ArgumentParserWithDefaults, CLIArgs

logger = logging.getLogger(__name__)

class SympolArgumentParser(ArgumentParserWithDefaults, Tap, CLIArgs):
    # this will call ArgumentParserWithDefaults.parse_unknown_args which sets explicit args
    def parse_args(self, args=None, *, known_only=False, **kwargs) -> CLIArgs:
        return Tap.parse_args(self, args, known_only=known_only, **kwargs)  # type: ignore[return-type]

    def configure(self) -> None:
        self.add_argument(
            "--no-adamW",
            action="store_false",
            dest="adamW",
            help="Do not use AdamW optimizer (explicitly sets to False)",
            required=False,
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

    def process_args(self) -> None:
        explicit_args_corrected = []
        for some_arg in self._explicit_args:
            if "no-" in some_arg:
                explicit_args_corrected.append("".join(some_arg.split("no-")))
            else:
                explicit_args_corrected.append(some_arg)
        explicit_arg_values = {arg: getattr(self, arg) for arg in explicit_args_corrected}

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
                    self.__dict__.update(best_cfg)
                    break
                if name == "-".join(self.env_id.lower().split("-")[:-1]):
                    if self.actor == "stateActionDT":
                        best_cfg = value["mlp"]
                    elif self.actor == "d-sdt":
                        best_cfg = value["sdt"]
                    else:
                        best_cfg = value[self.actor]
                    self.__dict__.update(best_cfg)
                    break
        if self.overwrite_explicit:
            self.__dict__.update(explicit_arg_values)
