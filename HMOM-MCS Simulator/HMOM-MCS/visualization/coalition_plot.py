# ================== FILE: visualization/coalition_plot.py ==================

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def plot_coalitions(coalitions, users, time_slot):
    """Visualize coalition formation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Coalition Formation at Time Slot {time_slot}', fontsize=14)

    # Coalition Sizes
    ax = axes[0]
    coalition_sizes = [len(c) for c in coalitions]
    ax.bar(range(len(coalition_sizes)), coalition_sizes, color='steelblue', edgecolor='black')
    ax.set_title('Coalition Sizes')
    ax.set_xlabel('Coalition ID')
    ax.set_ylabel('Number of Members')
    ax.grid(True, alpha=0.3, axis='y')

    # Coalition Spatial Distribution
    ax = axes[1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(coalitions)))

    for idx, coalition in enumerate(coalitions):
        if coalition:
            xs = [u.location[0] for u in coalition]
            ys = [u.location[1] for u in coalition]
            ax.scatter(xs, ys, c=[colors[idx]], s=50, alpha=0.7,
                       label=f'Coalition {idx}' if idx < 5 else '')

            # Draw convex hull
            if len(coalition) > 2:
                from scipy.spatial import ConvexHull
                points = np.array([[u.location[0], u.location[1]] for u in coalition])
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1],
                                color=colors[idx], alpha=0.3)
                except:
                    pass

    ax.set_title('Coalition Spatial Distribution')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    if len(coalitions) <= 5:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
