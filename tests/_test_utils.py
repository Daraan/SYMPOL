import sys
from unittest import mock

from typing_extensions import NotRequired, Required, get_origin, get_type_hints

fixed_args = mock.patch.object(sys, "argv", ["file.py", "--no-render_env"])
clean_args = mock.patch.object(sys, "argv", ["file.py"])
"""Use when comparing to CLIArgs"""


def get_explicit_required_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is Required}


def get_explicit_unrequired_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is NotRequired}


def get_required_keys(cls):
    return cls.__required_keys__ - get_explicit_unrequired_keys(cls)


def get_optional_keys(cls):
    return cls.__optional__keys - get_explicit_required_keys(cls)
