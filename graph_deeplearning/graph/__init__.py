import logging
import pickle
import os
from typing import Union, Iterable, Generator
from enum import Enum
from pyhocon import ConfigTree
from pathlib import Path


from graph_deeplearning.utilities import select, camelcase_to_lowercase, BaseConfig, DataStore
from graph_deeplearning.utilities.connection import Connection
from graph_deeplearning.graph.dse import DseConfig

# -------------------------------------------------------------------------------------------------
# 

class VertexError(TypeError):
    """Exception thrown when invalid vertex object is given as argument."""

    def __init__(self, value, types=None):
        if types:
            super().__init__("Graph vertices must be of type in {expected}, got '{actual}' instead".format(
                expected=repr([cls.__name__ for cls in types]),
                actual=type(value).__name__
            ))
        else:
            super().__init__("Invalid graph vertex type: {}".format(type(value).__name__))


class EdgeError(TypeError):
    """Exception thrown when invalid edge object is given as argument."""

    def __init__(self, value, types=None):
        if types:
            super().__init__("Graph edges must be of type in {expected}, got '{actual}' instead".format(
                expected=repr([cls.__name__ for cls in types]),
                actual=type(value).__name__
            ))
        else:
            super().__init__("Invalid graph edge type: {}".format(type(value).__name__))


class GraphMode(Enum):
    """Mode in which to open graph connection."""

    READ_WRITE = "rw"
    READ = "r"
    APPEND = "a"


# -------------------------------------------------------------------------------------------------
#

class GraphConfig(BaseConfig):
    """Configuration of graph database."""

    GROUP = "graph"

    __slots__ = [
        "backend",     # Default backend used for geo-image graph storage.
        "graph_name",  # Default name of geo-image graph.
        "dse_config"   # Configuration of DSE backend.
    ]

    def __init__(self, config: ConfigTree=None, group: str=None,
                 backend: DataStore=None, graph_name: str=None, dse_config: DseConfig=None,
                 log_level: Union[int, str]=None, verbosity: int=None):
        """Set configuration of geo-image graph interface."""
        super().__init__(config=config, group=group, log_level=log_level, verbosity=verbosity)
        self.dse_config = DseConfig.default().copy()
        GraphConfig.__update(self, config, group)
        self.backend = select(backend, self.backend)
        self.graph_name = select(graph_name, self.graph_name)
        self.dse_config = select(dse_config, self.dse_config)

    def __update(self, config: ConfigTree, group: str):
        """Update entries of **this** class only from given ConfigTree."""
        backend_config = self._get(config=config, group=group, key="backend", default=ConfigTree())
        if isinstance(backend_config, str):
            self.backend = DataStore.from_arg(backend_config)
        elif isinstance(backend_config, ConfigTree):
            self.backend = DataStore.from_arg(
                self._get(config=backend_config, group="", key="default", default=self.backend)
            )
            self.dse_config.update(
                config=self._get_config(
                    config=backend_config, group="", key=["dse", "dse_config"], default=ConfigTree()
                ),
                group=""
            )
        else:
            raise ValueError("geoimage.backend must be string or ConfigTree")
        self.graph_name = self._get_string(
            config=config, group=group, key=["graph_name", "graph"], default=self.graph_name
        )
        self.dse_config.graph_name = self.graph_name

    def update(self, config: ConfigTree, group: str=None):
        """Update configuration entries from given ConfigTree."""
        super().update(config, group)
        GraphConfig.__update(self, config, group)

# -------------------------------------------------------------------------------------------------
# Graph elements

class GraphElement(object):

    __slots__ = (
        "label"
    )

    def __init__(self, **kwargs):
        """Initialize graph element properties."""
        for name in self.__slots__:
            setattr(self, name, None)
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def properties(self) -> Iterable[str]:
        """Ordered list of property names."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")

    def defined(self) -> Iterable[str]:
        """Get list of defined (i.e., non-None) attributes."""
        return [name for name in self.__slots__ if getattr(self, name) is not None]  # pylint: disable=E1101        

    def get(self, name: str, default=None):
        """Get named property value or default value if not defined."""
        try:
            value = getattr(self, name)
        except AttributeError:
            value = None
        return default if value is None else value

    @classmethod
    def from_result(cls, result):
        """Initialize properties from graph query result."""
        return cls.from_properties(result.value)

    @classmethod
    def from_properties(cls, properties: dict):
        """Initialize properties from "properties" of graph query result."""
        kwargs = {}
        for name, value in properties.items():
            if isinstance(value, (list, tuple)):
                items = []
                for item in value:
                    if isinstance(item, dict) and "value" in item:
                        item = item["value"]
                    items.append(item)
                value = items
                if len(value) == 1:
                    value = value[0]
            elif isinstance(value, dict):
                if "value" in value:
                    value = value["value"]
            kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, props: dict, prefix: str=""):
        """Initialize properties from simplified value map of graph query result."""
        if prefix is None:
            prefix = ""
        for name, value in props.items():
            if name.startswith(prefix):
                name = name[len(prefix):]
            props[name] = value
        return cls(**props)

    def to_dict(self, prefix: str=""):
        """Get key/value map of graph element properties for the given schema."""
        if prefix is None:
            prefix = ""
        return dict([(prefix + name, getattr(self, name)) for name in self.defined()])

    def to_query_add(self, prefix: str="") -> str:
        """Get joined Gremlin ", 'name', name" statement of defined properties."""
        if prefix is None:
            prefix = ""
        return "".join([", '{1}', {0}{1}".format(prefix, name) for name in self.defined()])

    def to_query_has(self, prefix: str="") -> str:
        """Get joined Gremlin ".has('name', name)" statement of defined properties."""
        if prefix is None:
            prefix = ""
        return "".join([".has('{1}', {0}{1})".format(prefix, name) for name in self.defined()])

    def __repr__(self):
        """String representation for debug output."""
        return "{name}({kwargs})".format(
            name=self.__class__.__name__,
            kwargs=", ".join(["{k}={v}".format(k=name, v=repr(value)) for name, value in vars(self).items()])
        )


class Vertex(GraphElement):
    @staticmethod
    def _check_vertex(vertex):
        if not isinstance(vertex, 'Vertex'):
            raise TypeError("'vertex' argument must be of type Vertex")

    @classmethod
    def label(cls) -> str:
        """Vertex label."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")

    @classmethod
    def properties(cls) -> Iterable[str]:
        """Ordered list of property names."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")

    @classmethod
    def from_vertex(cls, vertex: 'Person'):
        """Initialize vertex properties from Person instance."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")

    def to_vertex(self):
        """Create Person given vertex properties."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")


class Edge(GraphElement):
    @classmethod
    def label(cls) -> str:
        """Edge label."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")

    @classmethod
    def properties(cls) -> Iterable[str]:
        """Get ordered list of field names, order must match expectation of ETL pipeline consumer."""
        raise NotImplementedError("Must be implemented by schema-specific subclass")

    @classmethod
    def _check_edge(cls, edge):
        if not isinstance(edge, 'Edge'):
            raise TypeError("'edge' argument must be of type Edge")


class Person(Vertex): pass
class Company(Vertex): pass
class Review(Edge): pass

# -------------------------------------------------------------------------------------------------
# Graph interface

class Graph(Connection):
    """Base class of graph data structures."""

    def __init__(self, name: str, mode: Union[GraphMode, str]=None, logger=None):
        """Initialize graph interface.

        Args:
            name: Name of graph.
            mode: Mode in which to establish server connections (if any).
            logger: Logger instance used to log messages. Use default logger if ``None``.
        """
        super().__init__()
        self.name = name or camelcase_to_lowercase(type(self).__name__)
        self.mode = GraphMode(select(mode, GraphMode.READ_WRITE))
        self.logger = logger or logging.getLogger("scape.graph")

    def set_mode(self, mode: Union[GraphMode, str]) -> 'Graph':
        """Change graph connection mode."""
        if mode is not None:
            mode = GraphMode(mode)
            if self.mode != mode:
                self.mode = mode
                self.reconnect()
        return self

    def is_temp(self) -> bool:
        """Whether graph is temporary and used for testing only."""
        return (
            self.name.startswith("scapetesting")
            or self.name.startswith("scape_testing")
            or self.name.startswith("test_")
            or self.name.startswith("testing_")
        )

    def exists(self) -> bool:
        """Whether this geo-image graph exists."""
        return True

    def empty(self) -> bool:
        """Whether graph is empty."""
        raise NotImplementedError("Must be implemented by subclass")

    def number_of_vertices(self) -> int:
        """Total number of vertices."""
        raise NotImplementedError("Must be implemented by subclass")

    def create(self, exist_ok=True) -> 'Graph':
        """Create graph structure."""
        if not exist_ok and not self.empty():
            raise Exception("Graph already exists")
        return self

    def clear(self, force: bool=False) -> 'Graph':
        """Drop vertices and edges."""
        raise NotImplementedError("Must be implemented by subclass")

    def drop(self, force: bool=False) -> 'Graph':
        """Drop graph structure."""
        return self.clear(force=force)

    # ----------------------------------------------------------------------------------------------
    # Pickling

    def __getstate__(self) -> dict:
        """Get object state as dictionary for pickling."""
        return {
            "name": self.name
        }

    def __setstate__(self, values: dict):
        """Set object state from unpickled dictionary."""
        self.name = values["name"]

    def dumps(self) -> bytes:
        """Serialize graph."""
        # return pickle.dumps(self)
        pass

    @classmethod
    def loads(cls, data: bytes) -> 'Graph':
        """Deserialize graph."""
        #return pickle.loads(data)
        pass

    def dump(self, path: Union[Path, str]):
        """Write graph to binary file."""
        pass

    @classmethod
    def load(cls, path: Union[Path, str]) -> 'Graph':
        """Load graph from binary file."""
        pass

    def set_path(self, path: Union[Path, str]) -> 'Graph':
        """Set path of pickle file used to make absolute local file paths relative and vice versa."""
        raise NotImplementedError("Reading/writing graph from/to pickle file not supported by subclass")

    def get_path(self) -> Union[Path, None]:
        """Get local file path of graph if any."""
        return None

    def save(self):
        """Save graph or do nothing if it is stored in database."""
        pass

    # ----------------------------------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: object) -> bool:
        """Compare graph to another."""
        if type(self) is not type(other):
            return False
        return self.name == other.name

    # ----------------------------------------------------------------------------------------------
    # Types

    @classmethod
    def _vertex_types(cls):
        """Subtypes of Vertex supported by this graph."""
        return (Vertex,)

    @classmethod
    def _edge_types(cls):
        """Subtypes of Edge supported by this graph."""
        return (Edge,)

    # ----------------------------------------------------------------------------------------------
    # Verification

    @classmethod
    def _valid_vertex(cls, arg: Vertex):
        """Check if argument is a valid vertex."""
        types = cls._vertex_types()
        if not isinstance(arg, types):
            raise VertexError(arg, types=types)
        return arg

    @classmethod
    def _valid_edge(cls, arg: Edge):
        """Check if argument is a valid edge."""
        types = cls._edge_types()
        if not isinstance(arg, types):
            raise EdgeError(arg, types=types)
        return arg

    # ----------------------------------------------------------------------------------------------
    # Insert vertex or edge

    def add_vertex(self, arg: Vertex) -> 'Graph':
        """Add a vertex to the graph, set additional properties on existing vertex, or modify its properties."""
        raise NotImplementedError("Must be implemented by subclass")

    def add_edge(self, arg: Edge) -> 'Graph':
        """Add an edge to the graph, set additional properties on an existing edge, or modify its properties."""
        raise NotImplementedError("Must be implemented by subclass")

    def add(self, *args: Union[Vertex, Edge]) -> 'Graph':
        """Convenience function to add either vertex or edge to graph."""
        for arg in args:
            if isinstance(arg, self._vertex_types()):
                self.add_vertex(arg)
            elif isinstance(arg, self._edge_types()):
                self.add_edge(arg)
            else:
                raise ValueError("Argument must be of a valid vertex or edge type, got: " + type(arg).__name__)
        return self

    # ----------------------------------------------------------------------------------------------
    # Find vertices or edges

    def find_vertices(self, *args: Vertex, select: Iterable[str]=None, limit: int=0) -> Generator[Vertex, None, int]:
        """Find vertices in the graph which share the specified subset of properties with the given vertex.

        Args:
            args: Sample vertex with properties of similar vertices to look for.
            select: Names of vertex properties of matching vertices to return.
            limit: When positive, generate at most the specified number of results.

        Returns:
            Generator of matching vertices whose return value is the total number of results.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def find_edges(self, *args: Edge, select: Iterable[str]=None, limit: int=0) -> Generator[Edge, None, int]:
        """Find edges in the graph which share the specified subset of properties with the given edge.

        Args:
            args: Sample edge with properties of similar edges to look for.
            select: Names of vertex properties of matching vertices to return.
            limit: When positive, generate at most the specified number of results.

        Returns:
            Generator of matching edges whose return value is the total number of results.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def find(self, *args: Union[Vertex, Edge], select: Iterable[str]=None, limit: int=0) \
            -> Generator[Union[Vertex, Edge], None, int]:
        """Convenience function to find either a vertex or an edge of the graph.

        Args:
            args: Sample vertex or edge with properties of similar graph elements to look for.
            select: Names of properties of matching graph elements to return.
            limit: When positive, generate at most the specified number of results.

        Returns:
            Generator of matching vertices or edges, respectively, whose return value is the total number of results.
        """
        edges = []
        verts = []
        for arg in args:
            if isinstance(arg, self._vertex_types()):
                verts.append(self._valid_vertex(arg))
            elif isinstance(arg, self._edge_types()):
                edges.append(self._valid_edge(arg))
            else:
                raise ValueError("Argument must be of valid vertex or edge type, got: " + type(arg).__name__)
        count = 0
        count += yield from self.find_vertices(*verts, select=select, limit=limit)
        count += yield from self.find_edges(*edges, select=select, limit=limit)
        return count

    def has_edge(self, edge: Edge) -> bool:
        """Find edges in the graph which share the specified subset of properties with the given edge."""
        try:
            next(self.find_edges(edge))
        except StopIteration:
            return False
        return True
