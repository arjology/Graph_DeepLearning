from enum import Enum
from typing import Iterable, Union

class GraphSchemaId(Enum):
    """Enumeration of graph schema IDs.

    These schema IDs are used for backwards compatibility. Enumeration entries are named
    after the year when the schema was first introduced, followed by a letter. It should
    not occur that we change the schema 26 times [a-z] in a single year.
    """
    v2019a = "v2019a"  # graph schema after first introduction of data abstraction layer start of 2019
    DEFAULT = v2019a


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


class GraphSchema(object):
    """Graph schema."""
    pass

def graph_schema(schema_id: GraphSchemaId=None):
    """Get graph schema for specified schema ID."""
    if schema_id is None:
        schema_id = GraphSchemaId.DEFAULT
    if schema_id == GraphSchemaId.v2019a:
        from schema.v2019a import GraphSchema_v2019a
        return GraphSchema_v2019a

    else:
        raise ValueError("Unknown graph schema: {0}".format(schema_id))
