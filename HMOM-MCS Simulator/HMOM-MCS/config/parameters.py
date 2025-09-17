# ================== FILE: config/parameters.py ==================

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SystemParameters:
    """System-wide parameters as defined in the paper"""
    # System scale
    n_users: int = 100  # Full: 1000
    n_tasks: int = 50  # Full: 500
    grid_size: Tuple[float, float] = (10.0, 10.0)  # km x km
    time_slots: int = 300  # Full: 500
    budget: float = 500.0  # Full: 5000.0

    # Thresholds
    theta_aoi: float = 2.0  # AoI threshold
    phi_energy: float = 50.0  # Energy limit
    psi_coverage: float = 0.8  # Coverage minimum
    omega_participation: float = 0.8  # Participation minimum

    # Weights for objective function
    lambda_energy: float = 0.01
    nu_malicious: float = 0.05
    kappa_quad: float = 0.01
    xi_aoi: float = 0.5
    pi_participation: float = 1.0
    zeta_pr_penalty: float = 1.0

    # User utility weights
    mu_social: float = 0.3
    beta_freshness: float = 0.4
    gamma_malicious: float = 0.1
    iota_lottery: float = 0.2
    vartheta_referral: float = 0.3

    # FL parameters
    epsilon_dp: float = 0.1
    delta_sensitivity: float = 2.0
    fl_learning_rate: float = 0.01
    fl_rounds: int = 10

    # BES-GO parameters
    population_size: int = 100
    max_generations: int = 200
    alpha_bes: float = 1.5
    beta_go: float = 0.8
    levy_beta: float = 1.5
    stagnation_threshold: int = 5
    crossover_rate: float = 0.5
    mutation_prob: float = 0.1
    mutation_sigma: float = 0.1

    # Dynamics
    rho_decay: float = 0.95
    alpha_energy_decay: float = 0.01

    # Participation mechanisms
    lottery_win_prob: float = 0.15
    lottery_base_bonus: float = 10.0
    referral_bonus: float = 5.0
    reputation_multiplier_factor: float = 0.1

    # Network
    social_graph_exponent: float = 2.5
    malicious_correlation: float = 0.05
