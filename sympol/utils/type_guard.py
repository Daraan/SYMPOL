from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from typing_extensions import LiteralString, TypeGuard, TypeIs

    from sympol.config_types.params_types import GeneralParams, MLPParams, SDTParams, SympolParams
    from sympol.mlp import Actor_MLP, Actor_MLP_Continuous
    from sympol.sdt import Actor_SDT
    from sympol.sympol import SYMPOL_RL


class _HasActor(Protocol):
    actor: Final[str]


def is_sympol_params(params: GeneralParams, args: _HasActor) -> TypeGuard[SympolParams]:  # noqa: ARG001
    """Equivalent of args.actor == "sympol"""
    return args.actor == "sympol"


def is_mlp_params(params: GeneralParams, args: _HasActor) -> TypeGuard[MLPParams]:  # noqa: ARG001
    """Equivalent of args.actor in ("mlp", "stateActionDT")"""
    return args.actor in ("mlp", "stateActionDT")


def is_sdt_params(params: GeneralParams, args: _HasActor) -> TypeGuard[SDTParams]:  # noqa: ARG001
    """Equivalent of args.actor in ("sdt", "d-sdt")"""
    return args.actor in ("sdt", "d-sdt")


def is_stateActionDT(actor, args: _HasActor) -> TypeGuard[Actor_MLP | Actor_MLP_Continuous]:  # noqa: ARG001
    return args.actor == "stateActionDT"


def is_d_sdt(actor, args: _HasActor) -> TypeGuard[Actor_SDT]:  # noqa: ARG001
    return args.actor == "d-sdt"


def is_sdt_actor(actor, args: _HasActor) -> TypeGuard[Actor_SDT]:  # noqa: ARG001
    return args.actor in {"sdt", "d-sdt"}


def is_sympol_actor(actor, args: _HasActor) -> TypeIs[SYMPOL_RL]:  # noqa: ARG001
    return args.actor == "sympol"
