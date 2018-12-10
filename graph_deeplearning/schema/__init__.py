from enum import Enum

class GraphSchemaId(Enum):
    """Enumeration of graph schema IDs.

    These schema IDs are used for backwards compatibility. Enumeration entries are named
    after the year when the schema was first introduced, followed by a letter. It should
    not occur that we change the schema 26 times [a-z] in a single year.
    """
    v2019a = "v2019a"  # graph schema after first introduction of data abstraction layer start of 2019
    DEFAULT = v2019a


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
