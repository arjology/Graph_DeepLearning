# -------------------------------------------------------------------------------------------------
# Connection classes
import logging
from enum import Enum
from graph_deeplearning.utilities import select, DseConfig

from dse.auth import PlainTextAuthProvider
from dse.cluster import Cluster, GraphExecutionProfile, EXEC_PROFILE_GRAPH_DEFAULT
from dse.cluster import GraphOptions
from dse.policies import AddressTranslator
from dse.graph import SimpleGraphStatement
from dse.policies import RoundRobinPolicy


class Connection(object):
    """Add nested session context capability to clients."""

    def __init__(self):
        """Initialize session state."""
        self._depth = 0

    def __enter__(self):
        """Establish connection when entering context."""
        self.connect()
        self._depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close connection when exiting outermost context."""
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self.close()

    def connect(self):
        """Establish connection if not connected."""
        return self

    def close(self):
        """Close connection if connected."""
        return self

    def shutdown(self):
        """Alias for close()."""
        return self.close()

    def reconnect(self):
        """Close and re-open connection if a connection is currently open."""
        if not self.is_closed():
            self.close()
            self.connect()

    def is_closed(self):
        """Whether session is closed."""
        # pylint: disable=R0201
        return False

    def is_open(self):
        """Whether session is open."""
        return not self.is_closed()


class CassandraEnvironment(Enum):
    """Which cassandra environment to use"""
    PROD = "PROD"
    DEV = "DEV"


class CqlClient(Connection):
    """Client functions for Cassandra database queries."""

    def __init__(self, keyspace: str=None, logger=None):
        """Initialize Cassandra database client."""
        super().__init__()
        self.keyspace = keyspace
        self.logger = logger or logging.getLogger("client:cql")

    def commit(self) -> 'CqlClient':
        """Commit changes."""
        return self

    def is_temp_database(self) -> bool:
        """Whether database is temporary and used for testing only."""
        return self.is_temp_keyspace()

    def is_temp_keyspace(self, keyspace: str=None) -> bool:
        """Check if keyspace is a temporary keyspace used for testing."""
        if keyspace is None:
            keyspace = self.keyspace
        if not keyspace:
            return False
        return (
            keyspace.startswith("scapetesting")
            or keyspace.startswith("scape_testing")
            or keyspace.startswith("test_")
        )

    def set_keyspace(self, keyspace: str) -> 'CqlClient':
        """Change default keyspace."""
        if keyspace is not None:
            self.keyspace = keyspace
        return self

    def has_table(self, name: str) -> bool:
        """Check if table exists."""
        raise NotImplementedError("Must be implemented by subclass")

    def drop_table(self, name: str, force: bool=False):
        """Drop table if it exists.

        Args:
            name: Name of table to drop.
            force: Force dropping a non-temporary testing table.
        """
        try:
            keyspace, name = name.split(".", 1)
        except ValueError:
            keyspace = self.keyspace
        if name and (force or self.is_temp_keyspace(keyspace)):
            self.execute("DROP TABLE IF EXISTS {table};".format(table=name))

    def drop_keyspace(self, keyspace: str=None, force: bool=False):
        """Drop keyspace if it exists.

        Args:
            keyspace: Name of keyspace to drop. Use ``self.keyspace`` if ``None``.
            force: Force dropping keyspace that does not start with a testing prefix.
                   Argument ``force=True`` is required for dropping other keyspaces
                   to minimise chances a non-testing keyspace is dropped by mistake.
        """
        if keyspace is None:
            keyspace = self.keyspace
        if keyspace and (force or self.is_temp_keyspace(keyspace)):
            self.execute("DROP KEYSPACE IF EXISTS {};".format(keyspace))

    def prepare(self, query):
        """Prepare statement for frequently executed query."""
        # pylint: disable=R0201
        return query

    def execute(self, query, values=None):
        """Execute CQL query."""
        raise NotImplementedError("Must be implemented by subclass")


class DseClient(CqlClient):
    """Datastax enterprise client for Cassandra database."""

    _instance = None  # global instance

    @classmethod
    def instance(cls) -> 'DseClient':
        """Get global database client using default configuration."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def connected(cls) -> 'DseClient':
        """Get connected global DSE client instance."""
        return cls.instance().connect()

    def __init__(self, config: DseConfig=None, keyspace: str=None, profiles=None, logger=None):
        """Initialize Datastax enterprise Cassandra client."""
        # lambda may not be required: pylint: disable=W0108
        self.config = select(config, lambda: DseConfig.default()).copy()
        if keyspace is not None:
            self.config.keyspace = keyspace
        if logger is None:
            logger = logging.getLogger("scape.client.cql.dse")
            logger.setLevel(self.config.log_level)
        if logger.level > logging.DEBUG:
            logging.getLogger('dse').setLevel(logging.WARNING)
            logging.getLogger('dse.cluster').setLevel(logging.WARNING)
            logging.getLogger('dse.protocol').setLevel(logging.ERROR)
        if profiles is None:
            profiles = {}
        if EXEC_PROFILE_DEFAULT not in profiles:
            profiles[EXEC_PROFILE_DEFAULT] = ExecutionProfile(load_balancing_policy=RoundRobinPolicy())
        self.profiles = profiles
        self.cluster = None
        self.session = None
        super().__init__(keyspace=self.config.keyspace, logger=logger)

    def connect(self) -> 'DseClient':
        """Open DSE session."""
        if self.is_closed():
            if self.cluster is None:
                self.logger.debug("Obtain DSE cluster instance")
                if self.config.addr_dict is None:
                    address_translator = None
                else:
                    address_translator = AddressTranslator(self.config.addr_dict)
                self.cluster = Cluster(
                    self.config.cluster,
                    address_translator=address_translator,
                    auth_provider=PlainTextAuthProvider(
                        username=self.config.username,
                        password=self.config.password
                    ),
                    execution_profiles=self.profiles
                )
            try:
                self.logger.debug("Establish connection with DSE cluster (keyspace='%s')", self.keyspace)
                self.session = self.cluster.connect(keyspace=self.keyspace)
            except NoHostAvailable as error:
                self.logger.error("Could not connect to DSE cluster")
                raise error
        return super().connect()

    def close(self) -> 'DseClient':
        """Close DSE session."""
        if self.cluster is not None:
            if not self.cluster.is_shutdown:
                self.logger.debug("Shutdown DSE cluster connection")
                self.cluster.shutdown()
            self.session = None
            self.cluster = None
        return super().close()

    def __del__(self):
        """Ensure DSE cluster is shutdown before destructing object."""
        self.close()

    def is_closed(self) -> bool:
        """Whether a DSE session is active or not."""
        return self.session is None or self.session.is_shutdown

    def create_keyspace(self, keyspace: str=None, exist_ok: bool=False,
                        replication: dict=None, replication_factor: int=1):
        """Create keyspace."""
        if keyspace is None:
            keyspace = self.keyspace
        if keyspace:
            query = "CREATE KEYSPACE "
            if exist_ok:
                query += " IF NOT EXISTS "
            query += keyspace
            if not replication:
                replication = {
                    'class': 'SimpleStrategy',
                    'replication_factor': replication_factor
                }
            query += " WITH REPLICATION = " + repr(replication)
            self.execute(query + ";")

    def set_keyspace(self, keyspace: str) -> 'DseClient':
        """Change default keyspace."""
        if self.session is None:
            raise RuntimeError("DSE connection must be established before default keyspace can be set")
        if keyspace is not None:
            self.session.set_keyspace(keyspace)
            self.config.keyspace = keyspace
            super().set_keyspace(keyspace)
        return self

    def has_table(self, name: str) -> bool:
        """Check if table exists."""
        return next(self.execute(
            "SELECT count(table_name) FROM system_schema.tables"
            " WHERE table_name='{table}' LIMIT 1 ALLOW FILTERING;".format(table=name)
        ))[0] == 1

    def prepare(self, query):
        """Prepare statement for frequently executed query."""
        if self.session is None:
            raise RuntimeError("DSE connection must be established before queries can be prepared")
        return self.session.prepare(query)

    def execute(self, query, values=None):
        """Execute CQL query."""
        if self.session is None:
            raise RuntimeError("DSE connection must be established before executing queries")
        if isinstance(query, PreparedStatement):
            statement = query.bind(values)
        elif isinstance(query, str):
            statement = SimpleStatement(query)
        else:
            statement = query
        self.logger.debug(repr(query))
        return iter(self.session.execute(statement, values))


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
