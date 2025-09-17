# ================== FILE: visualization/network_plot.py ==================

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_social_network(G: nx.Graph, users, title="Social Network Structure"):
    """Visualize social network with user attributes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14)

    # Network structure
    ax = axes[0]
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Node colors by credibility
    node_colors = [users[i].credibility if i < len(users) else 0.5
                   for i in range(G.number_of_nodes())]

    # Node sizes by social influence
    node_sizes = [users[i].social_influence * 50 if i < len(users) else 50
                  for i in range(G.number_of_nodes())]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, cmap='YlOrRd',
                           vmin=0, vmax=1, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    ax.set_title('Network Structure')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='User Credibility')

    # Degree distribution
    ax = axes[1]
    degrees = [G.degree(n) for n in G.nodes()]
    ax.hist(degrees, bins=20, edgecolor='black', alpha=0.7)
    ax.set_title('Degree Distribution')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of Nodes')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
