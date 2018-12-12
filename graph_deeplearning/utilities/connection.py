# -------------------------------------------------------------------------------------------------
# Connection classes
import logging
from enum import Enum

from dse.auth import PlainTextAuthProvider
from dse.cluster import GraphOptions, Cluster, ExecutionProfile, GraphExecutionProfile,\
    EXEC_PROFILE_DEFAULT, EXEC_PROFILE_GRAPH_DEFAULT, EXEC_PROFILE_GRAPH_SYSTEM_DEFAULT
from dse.auth import SaslAuthProvider    
from dse.policies import AddressTranslator
from dse.graph import SimpleGraphStatement
from dse.policies import RoundRobinPolicy


class AddressTranslate(AddressTranslator):
    def __init__(self, address_translation: dict):
        self.address_translation = address_translation

    def translate(self, addr: str) -> str:
        if addr in self.address_translation:
            return self.address_translation.get(addr)
        else:
            return addr


def dse_get_session(username: str,
                    password: str,
                    cluster_ip: list,
                    address_dict: dict = None,
                    graph_name: str=None,
                    profiles: dict=None
                    ):

    if username is None or password is None:
        ptap_kwargs = {
            "username": username,
            'password': password,
        }
        auth_provider = PlainTextAuthProvider(**ptap_kwargs)
    else:
        auth_provider = None
    cluster = None
    if graph_name is None:
        cluster = Cluster(contact_points=cluster_ip, auth_provider=auth_provider)
    else:
        ep = GraphExecutionProfile(
            request_timeout=120.0,
            load_balancing_policy=RoundRobinPolicy(),
            graph_options=GraphOptions(graph_name=graph_name)
        )
        print(cluster_ip)
        cluster = Cluster(
            contact_points=cluster_ip,
            auth_provider=auth_provider,
            execution_profiles={
                EXEC_PROFILE_GRAPH_DEFAULT: ep,
                EXEC_PROFILE_DEFAULT: ExecutionProfile(load_balancing_policy=RoundRobinPolicy())
            }
        )
    if cluster is not None:
        return cluster.connect()
    else:
        return None
