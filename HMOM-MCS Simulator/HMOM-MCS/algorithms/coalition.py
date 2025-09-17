# ================== FILE: algorithms/coalition.py ==================

import numpy as np
from typing import List


class CoalitionFormation:
    """Coalition formation via hedonic games at Edge layer"""

    def __init__(self, params):
        self.params = params
        self.coalitions = []

    def compute_marginal_utility(self, user, coalition: List, tasks: List, t: int) -> float:
        """Compute marginal utility for joining coalition"""
        if not coalition:
            return 0.0

        base_utility = sum([u.credibility for u in coalition]) / len(coalition)
        time_factor = self.params.rho_decay ** (t - user.availability[0])
        social_bonus = sum([user.social_influence * 0.1 for u in coalition])

        samples = []
        for _ in range(10):
            proc_time = np.random.uniform(0.2, 1.0)
            sample_utility = base_utility * time_factor + social_bonus - proc_time * 0.1
            samples.append(sample_utility)

        return np.mean(samples)

    def form_coalitions(self, users: List, tasks: List, t: int) -> List[List]:
        """Form coalitions using hedonic game"""
        coalitions = [[user] for user in users]
        improved = True
        iterations = 0
        max_iterations = 10

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for user in users:
                current_coalition = next((c for c in coalitions if user in c), None)
                if not current_coalition:
                    continue

                best_coalition = current_coalition
                best_utility = self.compute_marginal_utility(user, current_coalition, tasks, t)

                for coalition in coalitions:
                    if coalition == current_coalition:
                        continue

                    overlap = all(abs(user.availability[0] - u.availability[0]) < 5
                                  for u in coalition)
                    if not overlap:
                        continue

                    utility = self.compute_marginal_utility(user, coalition + [user], tasks, t)
                    if utility > best_utility:
                        best_utility = utility
                        best_coalition = coalition

                if best_coalition != current_coalition:
                    current_coalition.remove(user)
                    best_coalition.append(user)
                    improved = True
                    coalitions = [c for c in coalitions if c]

        return coalitions
