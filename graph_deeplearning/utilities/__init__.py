from typing import Union, Callable, Iterable, NamedTuple
from enum import Enum
from pyhocon import ConfigTree, ConfigFactory
from pathlib import Path
import logging
import os
from copy import copy as make_copy
import numpy as np
from argparse import ArgumentTypeError as err

# -------------------------------------------------------------------------------------------------
# Logging

def to_log_level(level: Union[str, int]):
    """Get logging level from log_level argument."""
    try:
        level = int(level)
    except ValueError:
        log_levels = ("CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "DEBUG", "INFO")
        if level.upper() not in log_levels:
            raise ValueError("Invalid logging level, must be either integer or one of {}".format(log_levels))
    return level


def get_log_level(config):
    """Get logging level from parsed module configuration."""
    return to_log_level(config.get_string("logging.level"))

# -------------------------------------------------------------------------------------------------
# Command argument parsing

ARG_CHOICES_TRUE = "true", "t", "yes", "y", "on", "1"
ARG_CHOICES_FALSE = "false", "f", "no", "n", "off", "0"
ARG_CHOICES_BOOL = ARG_CHOICES_TRUE + ARG_CHOICES_FALSE


def arg_to_bool(arg: Union[bool, str]) -> bool:
    """Convert string argument of boolean choices to boolean value."""
    if isinstance(arg, bool):
        return arg
    return arg.lower() in ARG_CHOICES_TRUE

def camelcase_to_lowercase(arg: str) -> str:
    """Convert class name in upper camel case to lowercase with underscores as word delimiters."""
    return re.sub(r'(.)([A-Z])', '\\1_\\2', arg).lower()

# -------------------------------------------------------------------------------------------------
# Optional map

def optmap(func, *args):
    """Map values using the provided function or functor if non-None, otherwise leave it None."""
    values = [None if arg is None else func(arg) for arg in args]
    return values[0] if len(args) == 1 else values


# -------------------------------------------------------------------------------------------------
# DSE select statement

def select(value: Union[None, object], default: Union[object, Callable[[], object]]):
    """Default value if value is ``None`` else value itself.

    Use a callable for the ``default`` argument to avoid unnecesary evaluation of default value.
    """
    if value is None:
        if callable(default):
            return default()
        return default
    return value

# -------------------------------------------------------------------------------------------------
# Default configuration

DEFAULT_CONFIG = None
DEFAULT_REGION = None

DEFAULT_NOT_SET = "DEFAULT_NOT_SET"   

RESOURCES_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'resources'))


def default_config(parent: Path, name: str) -> ConfigTree:
    """Load default configuration."""
    if isinstance(parent, str):
        parent = Path(parent)
    dflt_config = ConfigFactory.parse_file(str(parent.joinpath(name)))
    user_config = ConfigTree()
    if not arg_to_bool(os.environ.get("IGNORE_USER_CONFIG", False)):
        user_config_path = Path.home().joinpath("." + name)
        if user_config_path.exists():  # pylint: disable=E1101
            user_config = ConfigFactory.parse_file(str(user_config_path))
    return ConfigTree.merge_configs(dflt_config, user_config)


def load_default_config(path: Union[Path, str]=None):
    """Load custom configuration from specified file. Modifies global constants!"""
    # pylint: disable=global-statement
    global DEFAULT_CONFIG
    if path:
        DEFAULT_CONFIG = ConfigTree.merge_configs(DEFAULT_CONFIG, ConfigFactory.parse_file(str(path)))
    else:
        DEFAULT_CONFIG = default_config(RESOURCES_PATH, "graph.conf")

# Load default configuration, sets global constants declared above
load_default_config()

# -------------------------------------------------------------------------------------------------
# With this PathType class, you can simply specify the following argument type to match only an 
# existing directory--anything else will give an error message:

class PathType(object):
    def __init__(self, exists=True, type='file', dash_ok=True):
        '''exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
           type: file, dir, symlink, None, or a function returning True for valid paths
                None: don't care
           dash_ok: whether to allow "-" as stdin/stdout'''

        assert exists in (True, False, None)
        assert type in ('file','dir','symlink',None) or hasattr(type,'__call__')

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        if string=='-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise err('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise err('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise err('standard input/output (-) not allowed')
        else:
            e = os.path.exists(string)
            if self._exists==True:
                if not e:
                    raise err("path does not exist: '%s'" % string)

                if self._type is None:
                    pass
                elif self._type=='file':
                    if not os.path.isfile(string):
                        raise err("path is not a file: '%s'" % string)
                elif self._type=='symlink':
                    if not os.path.symlink(string):
                        raise err("path is not a symlink: '%s'" % string)
                elif self._type=='dir':
                    if not os.path.isdir(string):
                        raise err("path is not a directory: '%s'" % string)
                elif not self._type(string):
                    raise err("path not valid: '%s'" % string)
            else:
                if self._exists==False and e:
                    raise err("path exists: '%s'" % string)

                p = os.path.dirname(os.path.normpath(string)) or '.'
                if not os.path.isdir(p):
                    raise err("parent path is not a directory: '%s'" % p)
                elif not os.path.exists(p):
                    raise err("parent directory does not exist: '%s'" % p)

        return string