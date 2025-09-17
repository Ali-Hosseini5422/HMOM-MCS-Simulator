# ================== FILE: models/user.py ==================

import numpy as np


class User:
    """User entity with all attributes from Table 3"""

    def __init__(self, user_id: int, params):
        self.id = user_id
        self.location = np.random.uniform(0, params.grid_size[0], 2)
        self.energy = np.clip(np.random.normal(100, 20), 50, 200)
        self.credibility = np.random.uniform(0.5, 1.0)
        self.social_influence = np.clip(np.random.normal(5, 2), 1, 10)
        self.malicious_prob = 0.0
        self.participation_history = np.random.uniform(0.4, 0.9)
        self.availability = sorted(np.random.uniform(0, 20, 2))
        self.effort_sensing = 0.0
        self.effort_computation = 0.0
        self.utility = 0.0
        self.coalition = None
        self.received_payment = 0.0
        self.lottery_bonus = 0.0
        self.referral_bonus = 0.0
        self.malicious_pred = 0.0
        self.participation_pred = 0.0

    def update_participation_history(self, participated: bool, rho: float,
                                     delta: float, eta: float):
        """Update participation history with AR(1) dynamics"""
        noise = np.random.normal(0, 0.2)
        if participated:
            self.participation_history = (rho * self.participation_history +
                                          delta * self.social_influence / 10 +
                                          eta + noise)
            self.participation_history = np.clip(self.participation_history, 0, 1)
