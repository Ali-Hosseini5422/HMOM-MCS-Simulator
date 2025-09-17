# ================== FILE: visualization/blockchain_plot.py ==================
"""
visualization/blockchain_plot.py
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_blockchain(blockchain, time_slots):
    """Visualize blockchain statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Blockchain Performance Metrics', fontsize=14)

    if not blockchain.blocks:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No blocks mined yet',
                    ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        return fig

    # Blocks over time
    ax = axes[0, 0]
    block_times = list(range(len(blockchain.blocks)))
    ax.plot(block_times, range(1, len(blockchain.blocks) + 1),
            'b-', linewidth=2, marker='o')
    ax.set_title('Block Production')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Blocks')
    ax.grid(True, alpha=0.3)

    # Transactions per block
    ax = axes[0, 1]
    txs_per_block = [len(block['txs']) for block in blockchain.blocks]
    ax.bar(range(len(txs_per_block)), txs_per_block,
           color='green', edgecolor='black', alpha=0.7)
    ax.set_title('Transactions per Block')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Number of Transactions')
    ax.grid(True, alpha=0.3, axis='y')

    # Energy consumption
    ax = axes[1, 0]
    energy_per_block = [block['energy'] for block in blockchain.blocks]
    cumulative_energy = np.cumsum(energy_per_block)
    ax.plot(range(len(cumulative_energy)), cumulative_energy,
            'orange', linewidth=2)
    ax.set_title('Cumulative Energy Consumption')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Energy (kWh)')
    ax.grid(True, alpha=0.3)

    # Validator distribution
    ax = axes[1, 1]
    validators = [block['validator'] for block in blockchain.blocks]
    unique_validators, counts = np.unique(validators, return_counts=True)
    ax.pie(counts, labels=unique_validators, autopct='%1.1f%%', startangle=90)
    ax.set_title('Validator Distribution')

    plt.tight_layout()
    return fig
