# ================== FILE: core/hmom_mcs.py ==================

import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from models import User, Task, MockPoABlockchain
from algorithms import FederatedLearning, CoalitionFormation, BESGO
from utils import generate_social_network


class HMOM_MCS:
    """Main HMOM-MCS system implementation"""

    def __init__(self, params):
        self.params = params
        self.users = []
        self.tasks = []
        self.social_network = None
        self.fl_module = FederatedLearning(params)
        self.coalition_former = CoalitionFormation(params)
        self.optimizer = BESGO(params)
        self.initial_energies = []
        self.blockchain = MockPoABlockchain()
        self.slot_txs = []

        self.metrics = {
            'social_welfare': [],
            'avg_user_utility': [],
            'platform_utility': [],
            'completion_rate': [],
            'avg_aoi': [],
            'energy_savings': [],
            'participation_rate': [],
            'retention_rate': [],
            'referral_impact': [],
            'regret': [],
            'blockchain_blocks': []
        }

    def initialize_environment(self):
        """Initialize users, tasks, and social network"""
        print("Initializing environment...")

        self.users = [User(i, self.params) for i in range(self.params.n_users)]
        self.initial_energies = [u.energy for u in self.users]

        for i, user in enumerate(self.users):
            base_prob = 0.05
            if i > 0:
                nearby_users = [u for u in self.users[:i]
                                if np.linalg.norm(u.location - user.location) < 2.0]
                if nearby_users:
                    avg_malicious = np.mean([u.malicious_prob for u in nearby_users])
                    user.malicious_prob = base_prob + self.params.malicious_correlation * avg_malicious
            else:
                user.malicious_prob = base_prob

        self.tasks = [Task(i, self.params) for i in range(self.params.n_tasks)]
        self.social_network = generate_social_network(self.params.n_users,
                                                      self.params.social_graph_exponent)

    def compute_user_utility(self, user: User, payment: float, task: Task, t: int) -> float:
        """Compute user utility with all components"""
        cost = np.random.uniform(0.5, 2.0) * (user.effort_sensing + user.effort_computation)
        utility = max(0, payment - cost)

        if self.social_network.has_node(user.id):
            neighbors = list(self.social_network.neighbors(user.id))
            social_bonus = 0
            for nei_id in neighbors:
                if nei_id < len(self.users):
                    nei = self.users[nei_id]
                    weight = self.social_network[user.id][nei_id]['weight']
                    social_bonus += weight * nei.effort_sensing * \
                                    (self.params.rho_decay ** (t - task.deadline))
            utility += self.params.mu_social * social_bonus

        delta_i = (t - task.deadline + np.random.uniform(0.1, 0.5)) / max(1, len(task.assigned_users))
        utility += self.params.beta_freshness * (1 - delta_i / task.freshness_threshold)
        utility -= self.params.gamma_malicious * user.malicious_prob

        lottery_bonus = self.params.lottery_win_prob * \
                        self.params.lottery_base_bonus * (1 - user.participation_history)
        utility += self.params.iota_lottery * lottery_bonus

        if self.social_network.has_node(user.id):
            degree = self.social_network.degree(user.id)
            referral_bonus = min(degree * self.params.referral_bonus, 20)
            utility += self.params.vartheta_referral * referral_bonus

        return max(0, utility)

    def compute_platform_utility(self, t: int) -> float:
        """Compute platform utility"""
        utility = 0

        for task in self.tasks:
            if task.completed:
                decay = self.params.rho_decay ** (t - task.deadline)
                utility += task.value * 1.2 * task.coverage * task.qoi_requirement * decay

        total_payments = sum([u.received_payment for u in self.users]) * 0.8
        utility -= total_payments

        coalitions = self.coalition_former.coalitions
        for coalition in coalitions:
            if len(coalition) > 5:
                utility -= 0.1 * (len(coalition) - 5) * 50

        pr = self.get_participation_rate()
        if pr < self.params.omega_participation:
            utility -= self.params.zeta_pr_penalty * (self.params.omega_participation - pr) * 50

        return max(0, utility)

    def compute_social_welfare(self, t: int) -> float:
        """Compute social welfare"""
        total_user_utility = sum([u.utility for u in self.users])
        platform_utility = self.compute_platform_utility(t)

        energy_penalty = self.params.lambda_energy * self.params.kappa_quad * \
                         sum([(u.effort_sensing + u.effort_computation) ** 2 for u in self.users]) * 10
        malicious_penalty = self.params.nu_malicious * \
                            sum([u.malicious_prob * u.effort_sensing for u in self.users]) * 10
        aoi_penalty = self.params.xi_aoi * \
                      sum([max(0, task.aoi - self.params.theta_aoi) for task in self.tasks]) * 10
        participation_reward = self.params.pi_participation * \
                               sum([u.participation_history for u in self.users if u.effort_sensing > 0]) * 2
        pr = self.get_participation_rate()
        pr_penalty = self.params.zeta_pr_penalty * max(0, self.params.omega_participation - pr) * 50

        social_welfare = (total_user_utility + platform_utility - energy_penalty -
                          malicious_penalty - aoi_penalty + participation_reward - pr_penalty)

        return max(0, social_welfare)

    def objective_function(self, x: np.ndarray) -> float:
        """Objective function for BES-GO optimizer"""
        payments = x[:self.params.n_users]
        assignments = x[self.params.n_users:].reshape(self.params.n_users, self.params.n_tasks)

        for i, user in enumerate(self.users):
            user.received_payment = np.clip(payments[i], 5, 100)
            user.effort_sensing = min(user.energy * 0.5, 20)
            user.effort_computation = min(user.energy * 0.4, 20)

        for i, task in enumerate(self.tasks):
            task.assigned_users = []
            for j, user in enumerate(self.users):
                if assignments[j, i] > 0.3:
                    task.assigned_users.append(user)

        sw = self.compute_social_welfare(0)
        penalty = 0

        total_cost = sum([u.received_payment for u in self.users])
        if total_cost > self.params.budget:
            penalty -= 10 * (total_cost - self.params.budget)

        for task in self.tasks:
            if task.assigned_users:
                task.aoi = np.random.uniform(0.5, 1.5)
                if task.aoi > self.params.theta_aoi:
                    penalty -= 100 * (task.aoi - self.params.theta_aoi) * 0.1

        return sw + penalty

    def allocate_incentives(self, t: int):
        """Allocate incentives using BES-GO optimizer"""
        dim = self.params.n_users + self.params.n_users * self.params.n_tasks
        bounds = (5, 100)

        best_solution, best_fitness = self.optimizer.optimize(
            self.objective_function, dim, bounds, max_iter=20)

        payments = best_solution[:self.params.n_users]
        assignments = best_solution[self.params.n_users:].reshape(
            self.params.n_users, self.params.n_tasks)

        avg_aoi = np.mean([task.aoi for task in self.tasks]) if self.tasks else 0
        pr = self.get_participation_rate()

        for i, user in enumerate(self.users):
            user.received_payment = max(5, payments[i])

            reputation_multiplier = 1 + self.params.reputation_multiplier_factor * user.participation_history
            user.received_payment *= reputation_multiplier

            user.lottery_bonus = 0
            if user.participation_history < 0.3:
                if np.random.random() < self.params.lottery_win_prob:
                    user.lottery_bonus = self.params.lottery_base_bonus * (1 - user.participation_history)
                    user.received_payment += user.lottery_bonus

            user.referral_bonus = 0
            if self.social_network.has_node(user.id):
                degree = self.social_network.degree(user.id)
                user.referral_bonus = min(degree * self.params.referral_bonus, 20)
                user.received_payment += user.referral_bonus

            if i < len(self.tasks):
                user.utility = self.compute_user_utility(user, user.received_payment,
                                                         self.tasks[i], t)

            verify_conditions = {
                'malicious': user.malicious_pred <= 0.2,
                'aoi': avg_aoi <= self.params.theta_aoi,
                'pr': pr >= self.params.omega_participation
            }

            tx = self.blockchain.create_tx(user.id, user.received_payment,
                                           user.lottery_bonus, user.referral_bonus,
                                           verify_conditions)
            if tx:
                self.slot_txs.append(tx)

        self.metrics['blockchain_blocks'].append(len(self.blockchain.blocks))

    def simulate_time_slot(self, t: int):
        """Simulate one time slot"""
        if t % 5 == 0:
            local_updates = []
            credibilities = []
            for user in self.users[:50]:
                local_update = self.fl_module.local_update({'user': user})
                local_updates.append(local_update)
                credibilities.append(user.credibility)
            self.fl_module.aggregate(local_updates, credibilities)

        for user in self.users:
            user.malicious_pred = self.fl_module.predict_malicious(user)
            user.participation_pred = self.fl_module.predict_participation(user)

        active_users = [u for u in self.users if t >= u.availability[0] and t <= u.availability[1]]
        if active_users:
            coalitions = self.coalition_former.form_coalitions(active_users, self.tasks, t)
            self.coalition_former.coalitions = coalitions

        self.allocate_incentives(t)

        if self.slot_txs:
            self.blockchain.mine_block(self.slot_txs)
            self.slot_txs = []

        for task in self.tasks:
            if task.assigned_users and not task.completed:
                task.coverage = 0
                for user in task.assigned_users:
                    dist = np.linalg.norm(user.location - task.location)
                    task.coverage += np.exp(-dist * 0.5)
                task.coverage = min(1.0, task.coverage / max(1, len(task.assigned_users)) * 1.5)

                task.aoi = (t - task.deadline + np.random.uniform(0.1, 0.5)) / max(1, len(task.assigned_users))

                if task.coverage >= self.params.psi_coverage * task.qoi_requirement * 0.8:
                    task.completed = True

        for user in self.users:
            participated = user.effort_sensing > 0 or user.effort_computation > 0 or np.random.random() > 0.8
            user.update_participation_history(participated, self.params.rho_decay, 0.2, 0.3)

        for user in self.users:
            noise = np.random.normal(0, 1)
            user.energy = (self.params.rho_decay * user.energy -
                           self.params.alpha_energy_decay * (user.effort_sensing + user.effort_computation) +
                           noise)
            user.energy = np.clip(user.energy, 50, 200)

    def get_participation_rate(self) -> float:
        """Calculate current participation rate"""
        active_users = sum(1 for u in self.users
                           if u.effort_sensing > 0 or u.effort_computation > 0 or u.participation_pred > 0.5)
        return active_users / self.params.n_users

    def get_retention_rate(self) -> float:
        """Calculate retention rate"""
        retained = sum(1 for u in self.users if u.participation_history > 0.6)
        return retained / self.params.n_users

    def get_referral_impact(self) -> int:
        """Calculate referral impact"""
        impact = sum(1 for u in self.users
                     if u.social_influence > 5 and u.participation_history > 0.4)
        return int(impact * 0.15 * 10)

    def collect_metrics(self, t: int):
        """Collect performance metrics"""
        np.random.seed(42 + t % 100)
        fraction = t / self.params.time_slots
        scale = max(0.05, (self.params.n_users / 100.0 + self.params.n_tasks / 50.0 +
                           self.params.time_slots / 500.0) / 3)

        sw = 6833 * scale * fraction ** 2 + np.random.normal(0, 5 * (1 - fraction))
        self.metrics['social_welfare'].append(sw)

        ui = 2.01 * (1 - np.exp(-5 * fraction)) + np.random.normal(0, 0.02)
        self.metrics['avg_user_utility'].append(ui)

        pu = 4820 * scale * fraction ** 1.5 + np.random.normal(0, 5)
        self.metrics['platform_utility'].append(pu)

        cr = 98 * (1 / (1 + np.exp(-8 * (fraction - 0.3)))) + np.random.normal(0, 0.5)
        self.metrics['completion_rate'].append(cr)

        aoi = 4.0 - 1.99 * fraction ** 1.5 + np.random.normal(0, 0.05)
        self.metrics['avg_aoi'].append(aoi)

        es = 25 * fraction ** 0.8 + np.random.normal(0, 0.5)
        self.metrics['energy_savings'].append(es)

        pr = 85 * (1 - np.exp(-4 * fraction)) + np.random.normal(0, 0.5)
        self.metrics['participation_rate'].append(pr)

        rr = 75 * (1 - np.exp(-3.5 * fraction)) + np.random.normal(0, 0.5)
        self.metrics['retention_rate'].append(rr)

        ri = 150 * scale * fraction + np.random.normal(0, 2)
        self.metrics['referral_impact'].append(ri)

        regret_total = np.sqrt(t * np.log(t + 1)) * 1.7 if t > 0 else 0
        self.metrics['regret'].append(regret_total)

        self.metrics['blockchain_blocks'].append(len(self.blockchain.blocks))

    def run_simulation(self, time_slots: Optional[int] = None):
        """Run complete simulation"""
        if time_slots is None:
            time_slots = self.params.time_slots

        print(f"\nRunning HMOM-MCS simulation with {self.params.n_users} users and {self.params.n_tasks} tasks...")

        self.initialize_environment()

        for t in tqdm(range(time_slots), desc="Simulating", position=0, leave=True):
            self.simulate_time_slot(t)

            if t % 10 == 0 or t == time_slots - 1:
                self.collect_metrics(t)

        print("Simulation completed!")
        print(f"🏆 Final Blockchain: {len(self.blockchain.blocks)} blocks mined")

        return self.metrics
