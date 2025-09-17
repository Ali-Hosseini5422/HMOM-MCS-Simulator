# ================== FILE: algorithms/federated_learning.py ==================

import numpy as np
from typing import Dict, List


class FederatedLearning:
    """FL for predictions at Fog layer"""

    def __init__(self, params):
        self.params = params
        self.global_weights = np.random.randn(10)

    def add_dp_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Laplace noise for differential privacy"""
        sensitivity = self.params.delta_sensitivity
        scale = sensitivity / self.params.epsilon_dp
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    def local_update(self, user_data: Dict) -> np.ndarray:
        """Local model update at edge"""
        gradient = np.random.randn(10) * 0.1
        local_weights = self.global_weights - self.params.fl_learning_rate * gradient
        return self.add_dp_noise(local_weights)

    def aggregate(self, local_updates: List[np.ndarray], credibilities: List[float]):
        """Weighted aggregation at fog"""
        weights = np.array(credibilities) / sum(credibilities)
        self.global_weights = np.average(local_updates, axis=0, weights=weights)

    def predict_malicious(self, user) -> float:
        """Predict malicious probability"""
        features = np.array([
            user.credibility,
            user.participation_history,
            user.social_influence / 10
        ])
        score = 1 / (1 + np.exp(-np.dot(features, self.global_weights[:3])))
        return score

    def predict_participation(self, user) -> float:
        """Predict participation propensity"""
        features = np.array([
            user.participation_history,
            user.social_influence / 10,
            user.energy / 200
        ])
        score = 1 / (1 + np.exp(-np.dot(features, self.global_weights[3:6])))
        return score
