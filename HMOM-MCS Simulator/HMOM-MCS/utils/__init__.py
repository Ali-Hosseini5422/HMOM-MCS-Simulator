# ================== FILE: utils/__init__.py ==================

from .network_generator import generate_social_network

__all__ = ['generate_social_network']

# ================== FILE: utils/network_generator.py ==================
"""
utils/network_generator.py
"""
import networkx as nx
import numpy as np


def generate_social_network(n_users: int, exponent: float) -> nx.Graph:
    """Generate power-law social network as per paper"""
    G = nx.barabasi_albert_graph(n_users, m=3)

    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(0.1, 1.0)

    return G
