# ================== FILE: main.py ==================

import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List

# Import configurations
from config.parameters import SystemParameters

# Import core system
from core import HMOM_MCS

# Import visualization modules
from visualization import (
    plot_results,
    plot_comparison,
    plot_social_network,
    plot_distributions,
    plot_coalitions,
    plot_blockchain
)


def print_summary_statistics(results: Dict[str, List], params: SystemParameters):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("HMOM-MCS PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Configuration: {params.n_users} users, {params.n_tasks} tasks")
    print("-" * 80)

    last_n = min(3, len(results['social_welfare']))
    stats = {}

    for metric, values in results.items():
        if values:
            recent_values = values[-last_n:]
            stats[metric] = {
                'mean': np.mean(recent_values),
                'std': np.std(recent_values),
                'max': np.max(recent_values),
                'min': np.min(recent_values),
                'final': values[-1]
            }

    print(f"{'Metric':<30} {'Mean ± Std':<20} {'Final Value':<15} {'Range':<20}")
    print("-" * 80)

    metric_names = {
        'social_welfare': 'Social Welfare',
        'avg_user_utility': 'Avg User Utility',
        'platform_utility': 'Platform Utility',
        'completion_rate': 'Completion Rate (%)',
        'avg_aoi': 'Average AoI',
        'energy_savings': 'Energy Savings (%)',
        'participation_rate': 'Participation Rate (%)',
        'retention_rate': 'Retention Rate (%)',
        'referral_impact': 'Referral Impact',
        'regret': 'Cumulative Regret',
        'blockchain_blocks': 'Blockchain Blocks'
    }

    for key, name in metric_names.items():
        if key in stats:
            s = stats[key]
            mean_std = f"{s['mean']:.2f} ± {s['std']:.2f}"
            final = f"{s['final']:.2f}"
            range_str = f"[{s['min']:.2f}, {s['max']:.2f}]"
            print(f"{name:<30} {mean_std:<20} {final:<15} {range_str:<20}")

    print("=" * 80)


def run_scalability_analysis(base_params: SystemParameters):
    """Run scalability analysis"""
    print("\n" + "=" * 80)
    print("SCALABILITY ANALYSIS")
    print("=" * 80)

    configurations = [
        (5, 3, "Small"),
        (10, 5, "Medium"),
        (15, 8, "Large"),
        (20, 10, "Extra Large")
    ]

    all_results = {}

    for n_users, n_tasks, label in configurations:
        print(f"\nConfiguration: {label} - {n_users} users, {n_tasks} tasks")
        print("-" * 50)

        np.random.seed(42)

        params = SystemParameters()
        params.n_users = n_users
        params.n_tasks = n_tasks
        params.time_slots = 50

        system = HMOM_MCS(params)
        results = system.run_simulation(time_slots=50)

        all_results[label] = {
            'n_users': n_users,
            'n_tasks': n_tasks,
            'results': results
        }

        print(f"  Final Social Welfare: {results['social_welfare'][-1]:.2f}")
        print(f"  Final Participation Rate: {results['participation_rate'][-1]:.2f}%")
        print(f"  Final Completion Rate: {results['completion_rate'][-1]:.2f}%")
        print(f"  Final Average AoI: {results['avg_aoi'][-1]:.2f}")

    return all_results


def main():
    """Main function to run complete HMOM-MCS simulation"""
    print("=" * 80)
    print("HMOM-MCS: Hybrid Metaheuristic Driven Multilayer Incentive Mechanism")
    print("for Time-Variant Decentralized Mobile Crowdsensing")
    print("=" * 80)

    # Initialize system parameters
    params = SystemParameters()

    # Reset seed for base run
    np.random.seed(42)

    # Run base simulation
    print("\n1. Running base configuration simulation...")
    system = HMOM_MCS(params)
    results = system.run_simulation()

    # Print summary statistics
    print_summary_statistics(results, params)

    # Plot main results
    print("\n2. Generating visualizations...")

    # Main metrics plot
    fig1 = plot_results(results, params)
    plt.savefig('hmom_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Social network visualization
    fig2 = plot_social_network(system.social_network, system.users)
    plt.savefig('social_network.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Task and user distributions
    fig3 = plot_distributions(system.users, system.tasks, params)
    plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Coalition visualization (snapshot at time 50)
    if system.coalition_former.coalitions:
        fig4 = plot_coalitions(system.coalition_former.coalitions, system.users, 50)
        plt.savefig('coalitions.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Blockchain visualization
    fig5 = plot_blockchain(system.blockchain, params.time_slots)
    plt.savefig('blockchain.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Run scalability analysis
    print("\n3. Running scalability analysis...")
    scalability_results = run_scalability_analysis(params)

    # Plot scalability comparison
    fig6 = plot_comparison(scalability_results)
    plt.savefig('scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Save results to JSON
    print("\n4. Saving results...")
    json_results = {}
    for key, value in results.items():
        json_results[key] = [float(v) for v in value]

    with open('hmom_mcs_results.json', 'w') as f:
        json.dump({
            'base_configuration': {
                'n_users': params.n_users,
                'n_tasks': params.n_tasks,
                'results': json_results
            }
        }, f, indent=2)

    print("Results saved to hmom_mcs_results.json")
    print("Visualizations saved as PNG files")
    print("=" * 80)
    print("Simulation completed successfully!")

    return results, scalability_results


if __name__ == "__main__":
    # Run the main simulation
    results, scalability_results = main()