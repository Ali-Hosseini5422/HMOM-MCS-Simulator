# ================== FILE: visualization/metrics_plot.py ==================

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_results(results: Dict[str, List], params):
    """Plot all results as per paper figures"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'HMOM-MCS Performance Metrics (n={params.n_users}, m={params.n_tasks})', fontsize=16)

    time_points = list(range(0, len(results['social_welfare']) * 10, 10))

    # Social Welfare
    axes[0, 0].plot(time_points, results['social_welfare'], 'b-', linewidth=2)
    axes[0, 0].set_title('Social Welfare')
    axes[0, 0].set_xlabel('Time Slots')
    axes[0, 0].set_ylabel('SW')
    axes[0, 0].grid(True, alpha=0.3)

    # User and Platform Utility
    axes[0, 1].plot(time_points, results['avg_user_utility'], 'g-', label='Avg User Utility', linewidth=2)
    axes[0, 1].plot(time_points, results['platform_utility'], 'r-', label='Platform Utility', linewidth=2)
    axes[0, 1].set_title('Utilities')
    axes[0, 1].set_xlabel('Time Slots')
    axes[0, 1].set_ylabel('Utility')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Completion Rate
    axes[0, 2].plot(time_points, results['completion_rate'], 'c-', linewidth=2)
    axes[0, 2].set_title('Task Completion Rate')
    axes[0, 2].set_xlabel('Time Slots')
    axes[0, 2].set_ylabel('CR (%)')
    axes[0, 2].set_ylim([0, 105])
    axes[0, 2].grid(True, alpha=0.3)

    # Average AoI
    axes[1, 0].plot(time_points, results['avg_aoi'], 'm-', linewidth=2)
    axes[1, 0].axhline(y=params.theta_aoi, color='r', linestyle='--', label='Threshold')
    axes[1, 0].set_title('Average Age of Information')
    axes[1, 0].set_xlabel('Time Slots')
    axes[1, 0].set_ylabel('AoI')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Participation Rate
    axes[1, 1].plot(time_points, results['participation_rate'], 'b-', linewidth=2)
    axes[1, 1].axhline(y=params.omega_participation * 100, color='r', linestyle='--', label='Target')
    axes[1, 1].set_title('Participation Rate')
    axes[1, 1].set_xlabel('Time Slots')
    axes[1, 1].set_ylabel('PR (%)')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Retention Rate
    axes[1, 2].plot(time_points, results['retention_rate'], 'g-', linewidth=2)
    axes[1, 2].set_title('Retention Rate')
    axes[1, 2].set_xlabel('Time Slots')
    axes[1, 2].set_ylabel('RR (%)')
    axes[1, 2].set_ylim([0, 105])
    axes[1, 2].grid(True, alpha=0.3)

    # Energy Savings
    axes[2, 0].plot(time_points, results['energy_savings'], 'orange', linewidth=2)
    axes[2, 0].set_title('Energy Savings')
    axes[2, 0].set_xlabel('Time Slots')
    axes[2, 0].set_ylabel('Savings (%)')
    axes[2, 0].grid(True, alpha=0.3)

    # Referral Impact
    axes[2, 1].plot(time_points, results['referral_impact'], 'purple', linewidth=2)
    axes[2, 1].set_title('Referral Impact (New Users)')
    axes[2, 1].set_xlabel('Time Slots')
    axes[2, 1].set_ylabel('New Users')
    axes[2, 1].grid(True, alpha=0.3)

    # Cumulative Regret
    axes[2, 2].plot(time_points, results['regret'], 'r-', linewidth=2)
    axes[2, 2].set_title('Cumulative Regret (Sublinear)')
    axes[2, 2].set_xlabel('Time Slots')
    axes[2, 2].set_ylabel('Regret')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(all_results: Dict):
    """Plot comparison across different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Scalability Analysis - HMOM-MCS', fontsize=14)

    colors = ['blue', 'green', 'orange', 'red']

    for idx, (label, data) in enumerate(all_results.items()):
        time_points = list(range(0, len(data['results']['social_welfare']) * 10, 10))
        color = colors[idx % len(colors)]

        axes[0, 0].plot(time_points, data['results']['social_welfare'],
                        label=f"{label} (n={data['n_users']})", color=color, linewidth=2)
        axes[0, 1].plot(time_points, data['results']['participation_rate'],
                        label=label, color=color, linewidth=2)
        axes[1, 0].plot(time_points, data['results']['completion_rate'],
                        label=label, color=color, linewidth=2)
        axes[1, 1].plot(time_points, data['results']['avg_aoi'],
                        label=label, color=color, linewidth=2)

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title('Social Welfare Growth')
    axes[0, 1].set_title('Participation Rate')
    axes[1, 0].set_title('Completion Rate')
    axes[1, 1].set_title('Average AoI')

    plt.tight_layout()
    return fig