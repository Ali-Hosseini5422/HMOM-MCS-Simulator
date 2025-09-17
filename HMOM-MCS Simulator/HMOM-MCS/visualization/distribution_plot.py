# ================== FILE: visualization/distribution_plot.py ==================
"""
visualization/distribution_plot.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_distributions(users, tasks, params):
    """Visualize user and task spatial distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Spatial Distributions and Characteristics', fontsize=14)

    # User-Task Distribution
    ax = axes[0, 0]
    for user in users:
        ax.scatter(user.location[0], user.location[1],
                   c='blue', s=30, alpha=0.5, label='Users' if user.id == 0 else '')
    for task in tasks:
        ax.scatter(task.location[0], task.location[1],
                   c='red', s=50, marker='s', alpha=0.7,
                   label='Tasks' if task.id == 0 else '')
    ax.set_title('Spatial Distribution')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_xlim(0, params.grid_size[0])
    ax.set_ylim(0, params.grid_size[1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Task Coverage Areas
    ax = axes[0, 1]
    for task in tasks[:10]:  # First 10 tasks
        circle = Circle((task.location[0], task.location[1]),
                        radius=1.0, fill=False, edgecolor='red', alpha=0.3)
        ax.add_patch(circle)
        ax.text(task.location[0], task.location[1], f'T{task.id}',
                fontsize=8, ha='center')
    ax.set_title('Task Coverage Areas')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_xlim(0, params.grid_size[0])
    ax.set_ylim(0, params.grid_size[1])
    ax.grid(True, alpha=0.3)

    # User Energy Distribution
    ax = axes[1, 0]
    energies = [user.energy for user in users]
    ax.hist(energies, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(x=params.phi_energy, color='r', linestyle='--', label='Threshold')
    ax.set_title('User Energy Distribution')
    ax.set_xlabel('Energy Level')
    ax.set_ylabel('Number of Users')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Task Budget Distribution
    ax = axes[1, 1]
    budgets = [task.budget for task in tasks]
    ax.hist(budgets, bins=15, edgecolor='black', alpha=0.7, color='orange')
    ax.set_title('Task Budget Distribution')
    ax.set_xlabel('Budget')
    ax.set_ylabel('Number of Tasks')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig