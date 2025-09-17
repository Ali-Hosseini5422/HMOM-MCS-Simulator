# HMOM-MCS: Hybrid Metaheuristic Driven Multilayer Incentive Mechanism for Time-Variant Decentralized Mobile Crowdsensing

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of the HMOM-MCS framework, as described in the paper:

**HMOM-MCS: Hybrid Metaheuristic Driven Multilayer Incentive Mechanism for Time-Variant Decentralized Mobile Crowdsensing**  
*Authors: Seyed Ali Hosseini, Omid Sojoodi Shijani, Vahid Khajehvand*  
*Email: ali.hosseini5422@yahoo.com*  

The code is designed for reproducibility of the simulations and evaluations presented in the paper, including the hybrid Bald Eagle Search with Gazelle Optimizer (BES-GO), federated learning (FL) for privacy-preserving predictions, coalition formation via hedonic games, blockchain-based verification (mock PoA), and participation enhancement mechanisms (e.g., lotteries, social referrals, reputation multipliers). Experiments use real datasets (Rome taxi traces, Geolife, Foursquare) and synthetic data for scalability tests up to 2000 users.

## Key Features
- **Multilayer Architecture**: Mobile-edge-fog-cloud simulation with low-latency offloading (~50s end-to-end).
- **Hybrid Optimizer (BES-GO)**: Tailored for multi-objective MCS optimization (social welfare, AoI, participation rate >80%).
- **Privacy & Security**: Federated learning with ε-DP (ε=0.1) and mock Proof-of-Authority (PoA) blockchain for tamper-proof incentives.
- **Participation Enhancements**: Lottery-based entry rewards, social referral bonuses, and dynamic reputation multipliers to boost PR from <60% baseline to >85%.
- **Theoretical Guarantees**: Incentive compatibility, individual rationality, budget feasibility, and sublinear regret O(√(T log T)).
- **Evaluation Tools**: Comprehensive metrics (SW, CR, AoI, PR, RR, RI) with statistical tests (t-tests, ANOVA) and visualizations.
- **Scalability**: Handles n=1000 users/500 tasks by default; extensible to n=2000.

Results show 30%+ improvements in social welfare, task completion (98%), and participation over baselines (TD-DCBIM, IMELPT, RA-ABC).

## Installation
1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/hmom_mcs_project.git
   cd hmom_mcs_project
   ```

2. **Create a Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   The code requires Python 3.12.3+. Install via pip:
   ```
   pip install -r requirements.txt
   ```
   Core libraries:
   - `numpy`, `scipy`, `pandas` (data processing)
   - `torch` (federated learning)
   - `networkx` (social graphs)
   - `matplotlib`, `seaborn` (visualizations)
   - `PuLP` (optimization constraints)
   - `sympy` (symbolic derivations)
   - `tqdm` (progress bars)

   `requirements.txt` is included for exact versions. No internet access needed during runtime (per tool constraints).

4. **Download Datasets**:
   - Place datasets in `data/` (create if needed):
     - Rome taxi traces: [Download](https://www.dis.uniroma1.it/challenge9/download.shtml) → preprocess for 10km×10km grid.
     - Geolife: [Download](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) → filter for energy/mobility.
     - Foursquare: [Download](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) → sample for power-law graphs.
   - Synthetics generated on-the-fly via `utils/network_generator.py`.

## Usage
Run simulations via `main.py`. It initializes the HMOM-MCS system, runs T=500 time slots, optimizes via BES-GO, and outputs metrics/visualizations.

### Basic Run
```
python main.py --n_users 1000 --n_tasks 500 --T 500 --malicious_ratio 0.1 --output_dir results/
```
- `--n_users`: Number of users (default: 1000).
- `--n_tasks`: Number of tasks (default: 500).
- `--T`: Time slots (default: 500).
- `--malicious_ratio`: Fraction of malicious users (default: 0.1, correlated).
- `--output_dir`: Save results/plots (default: `results/`).

This reproduces base scenario results (e.g., SW=6833±150, PR=85±4%).

### Custom Scenarios
- **Scalability Test** (n=2000):
  ```
  python main.py --n_users 2000 --output_dir results_scalability/
  ```
- **Sensitivity Analysis** (vary budget B):
  ```
  python main.py --budget 1000 --output_dir results_budget_low/
  python main.py --budget 10000 --output_dir results_budget_high/
  ```
- **Ablation** (disable lotteries):
  ```
  python main.py --enable_lotteries False --output_dir results_no_lotteries/
  ```

### Outputs
- **Metrics**: CSV files (e.g., `metrics_base.csv`) with SW, u_i, U_P, CR, AoI, PR, RR, RI, regret. Includes stats (t-tests, p-values, Cohen's d).
- **Visualizations**: PNGs in `output_dir/`:
  - `metrics_plot.png`: Bar/line plots for KPIs.
  - `network_plot.png`: Social graph (power-law).
  - `distribution_plot.png`: User/task locations (10km grid).
  - `coalition_plot.png`: Hedonic coalitions.
  - `blockchain_plot.png`: Transaction logs (PoA blocks).
- **Logs**: Console/ `system.log` for convergence (e.g., BES-GO iterations).

To reproduce paper figures (e.g., Fig. 1: Regret vs. T):
```
python visualization/metrics_plot.py --input_dir results/ --fig regret --savefig True
```

## Directory Structure
```
hmom_mcs_project/
├── config/
│   └── parameters.py          # Configurable params (e.g., ρ=0.9, ε=0.1, B=5000)
├── models/
│   ├── __init__.py
│   ├── user.py                # User class (location, energy, ph_i, soc_i)
│   ├── task.py                # Task class (l_j, q_j, d_j, pq_j)
│   └── blockchain.py          # Mock PoA blockchain (consensus, smart contracts)
├── algorithms/
│   ├── __init__.py
│   ├── federated_learning.py  # FL module (weighted aggregation, DP noise)
│   ├── coalition.py           # Hedonic games for coalitions, Gale-Shapley matching
│   └── optimizer.py           # BES-GO hybrid (Levy flights, crossover, mutation)
├── visualization/
│   ├── __init__.py
│   ├── metrics_plot.py        # KPIs (SW, PR, AoI) bars/lines
│   ├── network_plot.py        # Social graph visualization
│   ├── distribution_plot.py   # Spatial distributions
│   ├── coalition_plot.py      # Coalition formations
│   └── blockchain_plot.py     # Transaction timelines
├── core/
│   ├── __init__.py
│   └── hmom_mcs.py            # Main HMOM-MCS class (layers, stages)
├── utils/
│   ├── __init__.py
│   └── network_generator.py   # Generate scale-free graphs (exponent=2.5)
└── main.py                    # Entry point for simulations
```

## Reproducing Paper Results
- **Base Scenario (Table I)**: `python main.py` → Compare `metrics_base.csv` to paper values (e.g., CR=98±2%).
- **Ablations (Table VI)**: Use flags like `--enable_lotteries False` → Expect PR drop of -25±3%.
- **Sensitivity (Table V)**: Vary weights in `config/parameters.py` (e.g., λ±20%) → Run and plot.
- **Stats**: Built-in t-tests/ANOVA in `metrics_plot.py` (p<0.001, d>1.5).
- **Full Repo**: 30 runs averaged; seed=42 for reproducibility.

For custom extensions (e.g., real blockchain integration), modify `models/blockchain.py`.

## License

## Acknowledgments
- Datasets: Rome taxi (University of Rome)
- Baselines re-implemented from cited papers.
- Simulations on Intel i7-12700H (32GB RAM); ~35s per run.

## Citation
If you use this code, please cite the paper:
```
@article{hosseini2025hmom,
  title={HMOM-MCS: Hybrid Metaheuristic Driven Multilayer Incentive Mechanism for Time-Variant Decentralized Mobile Crowdsensing},
  author={Hosseini, Seyed Ali and Sojoodi Shijani, Omid and Khajehvand, Vahid},
  journal=----------------------------------------------------------------,
  year={2025},
}
```

For issues/bugs: Open a GitHub issue. Contributions welcome via PRs!
