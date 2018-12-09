from typing import Union, Callable, Iterable, NamedTuple
from enum import Enum
from pyhocon import ConfigTree, ConfigFactory
from pathlib import Path
import logging
import os
from copy import copy as make_copy
import numpy as np

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

RESOURCES_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '.', 'resources'))


def default_config(parent: Path, name: str) -> ConfigTree:
    """Load default configuration."""
    if isinstance(parent, str):
        parent = Path(parent)
    dflt_config = ConfigFactory.parse_file(str(parent.joinpath(name)))
    user_config = ConfigTree()
    if not arg_to_bool(os.environ.get("SCAPE_IGNORE_USER_CONFIG", False)):
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
# Data store enumration

class DataStore(Enum):
    """Supported database backends."""

    UNKNOWN = -1      # Unknown/invalid backend
    DEFAULT = 0       # Default database backend

    DSE = 30          # Datastax enterprise server

    @classmethod
    def from_str(cls, value: str) -> Enum:
        """Map string to enumeration value."""
        if value == "DSE":
            return cls.DSE
        if value == "DEFAULT":
            return cls.DEFAULT
        return cls.UNKNOWN

    @classmethod
    def from_arg(cls, value: Union[Enum, str]):
        """From function argument."""
        if isinstance(value, str):
            backend = cls.from_str(value)
        elif isinstance(value, cls):
            backend = value
        else:
            raise TypeError("Invalid database/storage backend")
        return backend


class BaseConfig(object):
    """Base class of module configurations."""

    GROUP = ""

    __slots__ = [
        "log_level",  # Logging level of this component.
        "verbosity"   # Verbosity of logging messages.
    ]

    _default = dict()

    @classmethod
    def default(cls):
        """Get global configuration."""
        if cls not in BaseConfig._default:
            BaseConfig._default[cls] = cls()
        return BaseConfig._default[cls]

    def slots(self):
        """Get all __slots__ of this object."""
        return [name for slots in [getattr(cls, '__slots__', []) for cls in type(self).__mro__] for name in slots]

    def __init__(self, config: ConfigTree=None, group: str=None, log_level: Union[int, str]=None, verbosity: int=None):
        """Set common configuration entries."""
        for name in self.slots():
            setattr(self, name, None)
        self.log_level = logging.getLogger().level
        self.verbosity = 1
        BaseConfig.__update(self, config, group=group)
        BaseConfig.__update(self, config, group="default")
        self.log_level = to_log_level(select(log_level, self.log_level))
        self.verbosity = select(verbosity, self.verbosity)

    def __update(self, config: ConfigTree, group: str):
        """Update entries of **this** class only from given ConfigTree."""
        self.log_level = to_log_level(self._get(
            config=config, group=group, key="log_level", default=self.log_level, inherit=True
        ))
        self.verbosity = self._get_int(
            config=config, group=group, key="verbosity", default=self.verbosity, inherit=True
        )

    def update(self, config: ConfigTree, group: str=None):
        """Update configuration entries from given ConfigTree."""
        BaseConfig.__update(self, config, group)

    def copy(self):
        """Make a copy of this configuration."""
        config = make_copy(self)
        for name in self.slots():
            value = getattr(config, name)
            if isinstance(value, (list, tuple, dict)):
                setattr(config, name, make_copy(value))
        return config

    @staticmethod
    def _join(*args: str) -> str:
        """Join parts of configuration key, ignoring empty parts."""
        key = ''
        for arg in args:
            if arg.startswith("."):
                arg = arg[1:]
                key = ''
            if arg:
                if key:
                    key += '.'
                key += arg
        return key

    @staticmethod
    def _envvar(key: str) -> str:
        """Get name of environment variable from configuration entry key."""
        return "SCAPE_" + key.replace(".", "_").upper()

    @staticmethod
    def _parent(key: str) -> str:
        """Get parent group of configuration key."""
        return ".".join(key.split(".")[0:-1])

    @classmethod
    def _get(cls, key: Union[str, Iterable[str]], default: object=DEFAULT_NOT_SET,
             group: str=None, config: ConfigTree=None, inherit: bool=False, envvar: str=None) -> object:
        # too many branches: pylint: disable=R0912
        """Get value from default configuration using the specified getter."""
        config = select(config, DEFAULT_CONFIG)
        group = select(group, cls.GROUP)
        if isinstance(key, str):
            key = [key]
        value = None
        for name in key:
            config_key = cls._join(group, name)
            # 1. Consider environment variable
            if envvar != '':
                value = os.environ.get(select(envvar, cls._envvar(config_key)), None)
                if value is not None:
                    break
            # 2. Consider configuration entry of DEFAULT_CONFIG
            value = config.get(config_key, None)
            if value is not None:
                break
            # 3. Consider environment variables and/or configuration entries of parent group
            if inherit and value is None:
                parent_group = cls._parent(group)
                default_keys = [
                    cls._join(parent_group, name),
                    cls._join(parent_group, "default", name),
                    cls._join("default", group, name)
                ]
                if envvar != '':
                    for default_key in default_keys:
                        value = os.environ.get(cls._envvar(default_key), None)
                        if value is not None:
                            break
                if value is None:
                    for default_key in default_keys:
                        value = config.get(default_key, None)
                        if value is not None:
                            break
        # 4. Use hard-coded default value if specified (prefer to specify in DEFAULT_CONFIG)
        value = select(value, default)
        if value == DEFAULT_NOT_SET:
            raise KeyError("Missing configuration value for: " + repr([cls._join(group, name) for name in key]))
        return value

    @classmethod
    def _get_config(cls, key: Union[str, Iterable[str]], default: str=DEFAULT_NOT_SET,
                    group: str=None, config: ConfigTree=None, inherit: bool=False) -> str:
        """Get value from default configuration."""
        value = cls._get(config=config, group=group, key=key, default=default, inherit=inherit)
        if not isinstance(value, ConfigTree):
            raise ValueError("Entry group={} key={} must be a dictionary (ConfigTree)".format(group, key))
        return value

    @classmethod
    def _get_bool(cls, key: Union[str, Iterable[str]], default: bool=DEFAULT_NOT_SET,
                  group: str=None, config: ConfigTree=None, inherit: bool=False) -> bool:
        """Get value from default configuration."""
        return arg_to_bool(cls._get(config=config, group=group, key=key, default=default, inherit=inherit))

    @classmethod
    def _get_int(cls, key: Union[str, Iterable[str]], default: int=DEFAULT_NOT_SET,
                 group: str=None, config: ConfigTree=None, inherit: bool=False) -> int:
        """Get value from default configuration."""
        return int(cls._get(config=config, group=group, key=key, default=default, inherit=inherit))

    @classmethod
    def _get_float(cls, key: Union[str, Iterable[str]], default: float=DEFAULT_NOT_SET,
                   group: str=None, config: ConfigTree=None, inherit: bool=False) -> float:
        """Get value from default configuration."""
        return float(cls._get(config=config, group=group, key=key, default=default, inherit=inherit))

    @classmethod
    def _get_string(cls, key: Union[str, Iterable[str]], default: str=DEFAULT_NOT_SET,
                    group: str=None, config: ConfigTree=None, inherit: bool=False) -> str:
        """Get value from default configuration."""
        return str(cls._get(config=config, group=group, key=key, default=default, inherit=inherit))

    @classmethod
    def _get_list(cls, key: Union[str, Iterable[str]], default: list=DEFAULT_NOT_SET,
                  group: str=None, config: ConfigTree=None, inherit: bool=False) -> list:
        """Get value from default configuration."""
        value = cls._get(config=config, group=group, key=key, default=default, inherit=inherit)
        if isinstance(value, str):
            value = value.strip()
            if not value.startswith("[") or not value.endswith("]"):
                config_key = ' or '.join([cls._join(group, k) for k in key])
                raise ValueError("Value of {} must be comma-separated list".format(config_key))
            value = [item.strip() for item in value.split(",")]
        elif not isinstance(value, list):
            config_key = cls._join(group, key)
            raise ValueError("Value of {} must be list or string".format(config_key))
        return value

    def __getitem__(self, name: str) -> object:
        """Get named configuration entry."""
        obj = self
        for part in name.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise KeyError("Invalid attribute: " + name)
        return obj

    def __setitem__(self, name: str, value: object):
        """Set named configuration entry."""
        obj = self
        parts = name.split(".")
        for part in parts[0:-1]:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise KeyError("Invalid attribute: " + name)
        try:
            setattr(obj, parts[-1], value)
        except AttributeError:
            raise KeyError("Invalid attribute: " + name)

    def __repr__(self) -> str:
        """Convert to string representation."""
        s = type(self).__name__
        s += "("
        for i, slot in enumerate(self.slots()):
            if i > 0:
                s += ", "
            s += slot
            s += "="
            s += repr(getattr(self, slot))
        s += ")"
        return s

    def __eq__(self, other):
        """Compare configurations."""
        if not isinstance(other, self.__class__):
            return False
        for slot in self.slots():
            if getattr(self, slot) != getattr(other, slot):
                return False
        return True    


class GraphConfig(BaseConfig):
    """Configuration of graph database."""

    """Configuration of geo-image graph."""

    GROUP = "geoimage"

    __slots__ = [
        "backend",     # Default backend used for geo-image graph storage.
        "graph_name",  # Default name of geo-image graph.
    ]

    def __init__(self, config: ConfigTree=None, group: str=None,
                 backend: DataStore=None, graph_name: str=None,
                 log_level: Union[int, str]=None, verbosity: int=None):
        """Set configuration of geo-image graph interface."""
        super().__init__(config=config, group=group, log_level=log_level, verbosity=verbosity)
        self.backend = select(backend, self.backend)
        self.graph_name = select(graph_name, self.graph_name)

# -------------------------------------------------------------------------------------------------
# Person, Company, and Review named tuples
Person = NamedTuple('Person', [('name', str), ('gender', str), ('age', int), ('preferences', np.ndarray)])
Company = NamedTuple('Company', [('name', str), ('styles', np.ndarray)])
Review = NamedTuple('Review', [('name', str), ('company', 'str'), ('score', float)])

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
    