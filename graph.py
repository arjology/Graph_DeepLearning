import os
import logging
from enum import Enum
from typing import NamedTuple, Union, Dict, List, Iterable, Generator
from pyhocon import ConfigFactory, ConfigTree

from utils import GraphConfig, select
from schema import Person, Company, Review, Vertex, Edge, graph_schema, GraphSchemaId

from dse.auth import PlainTextAuthProvider
from dse.cluster import Cluster, GraphExecutionProfile, EXEC_PROFILE_GRAPH_DEFAULT
from dse.cluster import GraphOptions
from dse.policies import AddressTranslator
from dse.graph import SimpleGraphStatement

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.WARN)
log = logging.getLogger('python_shared-agent:dse')
log.setLevel(30)  # Set logging level to WARN


class AddressTranslate(AddressTranslator):
    def __init__(self, address_translation: dict):
        self.address_translation = address_translation

    def translate(self, addr: str) -> str:
        if addr in self.address_translation:
            return self.address_translation.get(addr)
        else:
            return addr


def dse_get_session(username: str, password: str, cluster_ip: list, address_dict: dict = None, graph_name: str=None):
    if address_dict is not None:
        address_translator = AddressTranslate(address_dict)
    else:
        address_translator = None

    if username is None or password is None:
        raise ValueError('\033[91m[ERROR] Authentication not provided')

    auth_provider = PlainTextAuthProvider(
        username=username, password=password)
    cluster = None
    if graph_name is None:
        cluster = Cluster(cluster_ip,
                          address_translator=address_translator,
                          auth_provider=auth_provider)
    else:
        ep = GraphExecutionProfile(graph_options=GraphOptions(graph_name=graph_name))
        cluster = Cluster(cluster_ip,
                          address_translator=address_translator,
                          auth_provider=auth_provider,
                          execution_profiles={EXEC_PROFILE_GRAPH_DEFAULT: ep})

    if cluster is not None:
        return cluster.connect()
    else:
        return None


class DsePropertyType(Enum):
    """DSE graph property key types."""

    Text = 1
    Int = 2
    Float = 3
    Double = 4
    Point = 5
    PointWithGeoBounds = 6
    Timestamp = 7


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


class DseGraphProperty(object):
    """ Definition of a graph property."""

    __slots__ = ["name", "type", "multiple"]

    def __init__(self, name: str, type: DsePropertyType, multiple: bool=False):
        """
        Args:
            name: name of the property
            type: type of the property (DsePropertyType)
            multiple: boolean on whether cardinality should be multiple
        """
        self.name = name
        self.type = type
        self.multiple = multiple

    def describe(self, exist_ok: bool=False) -> str:
        if self.type == DsePropertyType.PointWithGeoBounds:
            value_type = "Point().withGeoBounds"
        else:
            value_type = self.type.name
        return "schema.propertyKey('{name}').{value_type}().{multiplicity}()".format(
                    name=self.name,
                    value_type=value_type,
                    multiplicity="multiple" if self.multiple else "single"
                ) + (".ifNotExists()" if exist_ok else "") + ".create()"    


class CassandraEnvironment(Enum):
    """Which cassandra environment to use"""
    PROD = "PROD"
    DEV = "DEV"


class GraphMode(Enum):
    """Mode in which to open graph connection."""

    READ_WRITE = "rw"
    READ = "r"
    APPEND = "a"


class DseConfig(GraphConfig):
    """Configuration of Datastax enterprise client."""

    GROUP = "dse"

    __slots__ = (
        "cluster",     # IP addresses of cluster.
        "env",         # Flag for selecting production, test, or future (6.0.3) cluster
        "addr_dict",   # Optional map of cluster IPs to external IPs.
        "username",    # Database user name.
        "password",    # Database user password.
        "keyspace",    # Cassandra keyspace.
        "consumer",    # IP of bootstrap server of Kafka producer.
        "dse_config"   # Configuration of DSE backend.
    )

    def __init__(self, config: ConfigTree=None, group: str=None,
                 cluster: Union[List[str], str]=None, env: str=None, addr_dict: Dict[str, str]=None,
                 username: str=None, password: str=None, keyspace: str=None, graph_name: str=None,
                 log_level: Union[int, str]=None, verbosity: int=None):
        super().__init__(config=config, group=group, log_level=log_level, verbosity=verbosity)
        self.cluster = []
        self.env = None
        self.addr_dict = {}
        self.username = ""
        self.password = ""
        self.keyspace = ""
        self.graph_name = ""
        DseConfig.__update(self, config, "cassandra")
        DseConfig.__update(self, config, self._join("default", self.GROUP))
        DseConfig.__update(self, config, group)
        self.addr_dict = select(addr_dict, self.addr_dict)
        self.username = select(username, self.username)
        self.password = select(password, self.password)
        self.keyspace = select(keyspace, self.keyspace)
        self.graph_name = select(graph_name, self.graph_name)
        self.dse_config = DseConfig.default().copy()


    def __getattr__(self, key: str) -> object:
        """Get attribute using alternative name."""
        if key in ("ip", "cluster_ip"):
            return self.cluster
        if key == "ip_map":
            return self.addr_dict
        raise AttributeError("Object of type {} has no attribute named {}".format(type(self).__name__, key))

    def __update(self, config: ConfigTree, group: str):
        """Update entries of **this** class only from given ConfigTree."""
        self.env = CassandraEnvironment(self._get_string(config=config, group=group, key="env",
                                                         default=CassandraEnvironment.DEV.value))
        if self.env == CassandraEnvironment.DEV:
            self.cluster = self._get_list(config=config, group=group, key=["dev_cluster", "dev_ip"],
                                          default=self.cluster)
        self.addr_dict = self._get_addr_dict(config=config, group=group, key="addr_dict", default=self.addr_dict)
        self.username = self._get_string(config=config, group=group, key="username", default=self.username)
        self.password = self._get_string(config=config, group=group, key="password", default=self.password)
        self.keyspace = self._get_string(
            config=config, group=group, key="keyspace", default=self.keyspace, inherit=True
        )
        self.graph_name = self._get_string(
            config=config, group=group, key=["graph_name", "graph"], default=self.graph_name, inherit=True
        )
        self.consumer = self._get_string(
            config=config, group=group, key=["consumer", "broker"], default=self.consumer
        )

    def update(self, config: ConfigTree, group: str=None):
        """Update configuration entries from given ConfigTree."""
        super().update(config, group)
        DseConfig.__update(self, config, group)

    def _get_addr_dict(self, key: Union[str, Iterable[str]], group: str=None, config: ConfigTree=None,
                       default: Dict[str, str]=None) -> Dict[str, str]:
        """Get address translation dictionary for Cassandra."""
        addr_dict = {}
        for int_ip, ext_ip in self._get(config=config, group=group, key=key, default=default).items():
            addr_dict[int_ip.replace("\"", "")] = ext_ip
        return addr_dict


class DseGraph(object):
    """Geo-image graph stored in DSE graph database."""

    def __init__(self, name: str, mode: Union[GraphMode, str]=None,
                 config: DseConfig=None, schema: GraphSchemaId=None, logger=None):
        """Initialize DSE geo-image graph interface.

        Args:
            name: Name of graph. When client is not ``None``, this attribute is overridden by the name of
                  the graph that this ``client`` is connected to, i.e., this argument is ignored in this case.
            mode: Mode in which to open graph connection.
            config: DSE Config. Default configuration used if ``None``.
            schema: ID of graph schema to use. If ``None``, select based on graph ``name`` or assume default.
            logger: Logger used for debug, status, and other messages.
        """
        self.name = name
        self.logger = logging.getLogger("graph") if logger is None else logger
        self.mode = mode
        if schema is None:
            if self.name == "v2019a":
                schema = GraphSchemaId.v2019a
            else:
                schema = GraphSchemaId.DEFAULT
        config=DseConfig.default().dse_config if not config else config
        self.client = dse_get_session(username=config.username,
                                      password=config.password,
                                      cluster_ip=config.cluster_ip
                                    )                                    
        self.schema = graph_schema(schema)
        self.logger.debug("Using graph schema '%s'", self.schema.uid().value)
        self._queries = {}

    @classmethod
    def _vertex_types(cls):
        """Valid subtypes of Vertex supported by this graph."""
        return (Person, Company)

    @classmethod
    def _edge_types(cls):
        """Valid subtypes of Edge supported by this graph."""
        return (Review,)        

    def is_closed(self) -> bool:
        """Whether session is closed."""
        if self.mode == GraphMode.APPEND:
            return False
        return self.client.is_closed()

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
    # Upsert vertex or edge

    def add_person(self, *people: Person) -> 'DseGraph':
        """Upsert person."""
        for vertex in people:
            assert vertex.__class__ is Person
            assert self.mode == GraphMode.READ_WRITE or self.mode == GraphMode.APPEND
            # Check if required properties are set
            person = self.schema.person().from_vertex(vertex)
            # Push resources to remote storage
            self.logger.debug(
                "Commit modified resources of person (%s)",
                person.name # pylint: disable=E1101
            )
            # Add vertex to graph database
            self.logger.debug(
                "Add person label=%s, name=%s",
                person.label(), person.name  # pylint: disable=E1101
            )
            self.client.execute_graph(
                "graph.addVertex(label, lbl" + person.to_query_add() + ")",
                {
                    "lbl": person.label(),
                    **person.to_dict()
                }
            )
        return self

    def add_company(self, *companies: Company) -> 'DseGraph':
        """Upsert company."""
        for vertex in companies:
            assert vertex.__class__ is Company
            assert self.mode == GraphMode.READ_WRITE or self.mode == GraphMode.APPEND
            # Check if required properties are set
            company = self.schema.company().from_vertex(vertex)
            # Push resources to remote storage
            self.logger.debug(
                "Commit modified resources of company (%s)",
                company.name # pylint: disable=E1101
            )
            # Add vertex to graph database
            self.logger.debug(
                "Add company label=%s, name=%s",
                company.label(), company.name  # pylint: disable=E1101
            )
            self.client.execute_graph(
                "graph.addVertex(label, lbl" + company.to_query_add() + ")",
                {
                    "lbl": company.label(),
                    **company.to_dict()
                }
                )
        return self        

    def add_review(self, *edges: Review) -> 'DseGraph':
        """Upsert review from person to company."""
        # Person_* has no '*' member pylint: disable=E1101
        for edge in edges:
            assert edge.__class__ is Review
            assert self.mode == GraphMode.READ_WRITE or self.mode == GraphMode.APPEND
            # Check if required properties are set
            review = self.schema.review().from_edge(edge)
            if review.person is None or review.company is None:
                raise ValueError("'src' and 'dst' of review must be set")
            if not review.name.name:
                raise ValueError("Missing 'person' name of Review")
            if not review.company.name:
                raise ValueError("Missing 'company' name of Review")
            # Push resources to remote storage
            self.logger.debug(
                "Commit modified resources of review (%s, %s) <-> (%s, %s)",
                review.person.name,
                review.company.name
            )
            # Add edge to graph database
            self.logger.debug(
                "Add edge label=%s, person=(%s), company=(%s)",
                review.label(), review.person.name, review.company.name,
                review.dst.partition_key, review.dst.cluster_key
            )
            query = (
                    "src = graph.addVertex(label, src_lbl, name_lbl, person_name); "
                    "dst = graph.addVertex(label, dst_lbl, name_lbl, company_name); "
                    "src.addEdge(edge_lbl, dst" + review.to_query_add() + ")"
                )
            values = {
                    "name_label": self.schema.person().FIELDS.name.name,
                    "edge_lbl": review.label(),
                    "src_lbl": review.src.label(),
                    "person_name": review.person.name,
                    "dst_lbl": review.company.label(),
                    "company_name": review.company.name,
                    **review.to_dict()
                }
            self.client.execute_graph(query, values)
        return self        

    # ----------------------------------------------------------------------------------------------
    # Find vertices or edges

    @staticmethod
    def _select_statement(select: Iterable[str]) -> str:
        """Get Gremlin statement part to select values from ``.valueMap()``."""
        query = ".valueMap()"
        if select:
            query += ".select(" + ", ".join([repr(name) for name in select]) + ")"
        return query

    def find_people(self, *args: Person,
                    select: Iterable[str]=None, limit: int=0) -> Generator[Person, None, int]:
        """Find person vertices with given properties.

        Args:
            args: Vertex properties to search for.
            select: Optional subset of properties to return.
            limit: When positive, generate at most the specified number of results.

        Returns:
            Generator of matching vertices whose return value is the total number of results.
        """
        count = 0
        for vertex in args:
            if vertex.name and vertex.gender and vertex.age:
                limit = 1
            person = self.schema.person().from_vertex(vertex)
            query = (
                "g.V().hasLabel(lbl){props}.dedup()"
                ".order().by(name_lbl).order().by(age_lbl)"
                "{select}".format(
                    props=person.to_query_has(),
                    select=self._select_statement(select)
                )
            )
            if limit > 0:
                query += ".limit({})".format(limit)
            query += ".toList()"

            values = {
                "name_lbl": self.schema.person().FIELDS.name.name,
                "age_lbl": self.schema.person().FIELDS.age.name,
                "lbl": person.label(),
                **person.to_dict()
            }
            results = self.client.execute_graph(query, values)
            num_results = 0
            for result in results:
                num_results += 1
                yield self.schema.person().from_result(result).to_vertex()
            if num_results == 0:
                break
            count += num_results
        return count

    def find_companies(self, *args: Company,
                    select: Iterable[str]=None, limit: int=0) -> Generator[Person, None, int]:
        """Find company vertices with given properties.

        Args:
            args: Vertex properties to search for.
            select: Optional subset of properties to return.
            limit: When positive, generate at most the specified number of results.

        Returns:
            Generator of matching vertices whose return value is the total number of results.
        """
        count = 0
        for vertex in args:
            if vertex.name:
                limit = 1
            company = self.schema.company().from_vertex(vertex)
            query = (
                "g.V().hasLabel(lbl){props}.dedup()"
                ".order().by(name_lbl).order()"
                "{select}".format(
                    props=company.to_query_has(),
                    select=self._select_statement(select)
                )
            )
            if limit > 0:
                query += ".limit({})".format(limit)
            query += ".toList()"

            values = {
                "name_lbl": self.schema.company().FIELDS.name.name,
                "lbl": company.label(),
                **company.to_dict()
            }
            results = self.client.execute_graph(query, values)
            num_results = 0
            for result in results:
                num_results += 1
                yield self.schema.company().from_result(result).to_vertex()
            if num_results == 0:
                break
            count += num_results
        return count

    def find_reviews(self, *args:  Union[Person, Company, Review],
                     select: Iterable[str]=None, limit: int=0) -> Generator[Review, None, int]:
        """Find edges in graph corresponding to person review of company.

        Args:
            args: Example edge with properties of edges to look for.
                  To find an edge between two vertices, only set the ``src`` and ``dst`` vertices
                  of the edge and do not specify other properties. When only one vertex is specified,
                  all edges going in or out of this vertex are searched. Note that the edges are "undirected".
                  When a Person is given as argument, all reviews of this
                  person with other companies are generated.
                  When a Company is given as argument, all reviews of this
                  company with other persons are generated
            select: Names of edge properties to return (ignored).
            limit: Maximum number of results to generate.

        Returns:
            Generator of matching edges whose return value is the total number of results.
        """
        # unused argument 'select': pylint: disable=W0613
        count = 0
        for arg in args:
            edge = self._valid_edge(arg)
            review = self.schema.review().from_edge(edge)
            any_src = (not edge.person.name)
            any_dst = (not edge.company.name)
            if any_src and any_dst:
                start, other = None, None
            elif any_src:
                start, other = edge.company, None
            elif any_dst:
                start, other = edge.person, None
            else:
                start, other = edge.person, edge.company
            query = (
                "g.V().hasLabel(start){src_props}.as('src')"
                ".bothE(lbl){edge_props}.as('edge')"
                ".otherV().hasLabel(end){dst_props}.as('dst')"
            ).format(
                src_props=start.to_query_has("src_") if start else "",
                edge_props=review.to_query_has(),
                dst_props=other.to_query_has("dst_") if other else ""
            )
            query += ".select('edge')"
            query += ".order().by('~local_id')"
            query += ".select('edge', 'src', 'dst')"
            query += ".dedup()"
            if limit > 0:
                query += ".limit({0})".format(limit)
            query = SimpleGraphStatement(query)

            values = {
                "lbl": self.schema.review().label(),
                "start": start.label(),
                "other": other.label(),
                **review.to_dict()
            }
            if start:
                values.update(start.to_dict("src_"))
            if other:
                values.update(other.to_dict("dst_"))
            results = self.client.execute_graph(query, values)

            num_results = 0
            for result in results:
                num_results += 1
                yield self.schema.review().from_result(result).to_edge()
            if num_results == 0:
                break
            count += num_results
        return count

