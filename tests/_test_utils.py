import sys
from unittest import mock


fixed_args = mock.patch.object(sys, "argv", ["file.py", "--no-render_env"])
clean_args = mock.patch.object(sys, "argv", ["file.py"])
"""Use when comparing to CLIArgs"""
