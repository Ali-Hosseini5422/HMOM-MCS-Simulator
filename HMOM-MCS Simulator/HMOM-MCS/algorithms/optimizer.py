# ================== FILE: algorithms/optimizer.py ==================

import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional


class BESGO:
    """Hybrid Bald Eagle Search - Gazelle Optimizer"""

    def __init__(self, params):
        self.params = params
        self.population = None
        self.fitness = None
        self.gbest = None
        self.gbest_fitness = float('-inf')
        self.prey = None
        self.stagnation_count = 0
        self.generation = 0

    def levy_flight(self, beta: float = 1.5) -> float:
        """Generate Levy flight step"""
        sigma = (gamma_func(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma_func((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = 0.01 * (u / (abs(v) ** (1 / beta)))
        return step

    def initialize_population(self, dim: int, bounds: Tuple[float, float]):
        """Initialize population uniformly in feasible space"""
        self.population = np.random.uniform(bounds[0], bounds[1],
                                            (self.params.population_size, dim))
        self.fitness = np.zeros(self.params.population_size)

    def bes_exploration(self) -> np.ndarray:
        """BES exploration phase with Levy flights"""
        new_pop = np.copy(self.population)
        mean_loc = np.mean(self.population, axis=0)

        for i in range(self.params.population_size):
            r1, r2 = np.random.random(), np.random.random()
            new_pop[i] = self.population[i] + self.params.alpha_bes * r1 * (
                    mean_loc - np.abs(self.population[i]))
            levy_val = self.levy_flight(self.params.levy_beta) * np.sign(r2 - 0.5)
            new_pop[i] = new_pop[i] + levy_val * (self.gbest - self.population[i])

        return new_pop

    def go_exploitation(self) -> np.ndarray:
        """GO exploitation phase with adaptive beta"""
        new_pop = np.copy(self.population)
        adaptive_beta = self.params.beta_go * (1 - self.generation / self.params.max_generations)

        for i in range(self.params.population_size):
            r = np.random.random()
            new_pop[i] = self.population[i] + adaptive_beta * r * (self.prey - self.population[i])

        return new_pop

    def hybrid_crossover(self, pop1: np.ndarray, pop2: np.ndarray) -> np.ndarray:
        """Crossover between BES and GO populations"""
        n_elite = int(0.2 * self.params.population_size)
        idx1 = np.argsort(self.fitness)[-n_elite:]
        idx2 = np.argsort(self.fitness)[-n_elite:]
        elite1 = pop1[idx1]
        elite2 = pop2[idx2]

        offspring = np.empty_like(pop1[:n_elite])
        for i in range(n_elite):
            mask = np.random.random(elite1.shape[1]) < self.params.crossover_rate
            offspring[i] = np.where(mask, elite1[i % len(elite1)],
                                    elite2[i % len(elite2)])

        if np.random.random() < self.params.mutation_prob:
            mutation = np.random.normal(0, self.params.mutation_sigma, offspring.shape)
            offspring += mutation

        return offspring

    def optimize(self, objective_func, dim: int, bounds: Tuple[float, float],
                 max_iter: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Main optimization loop"""
        if max_iter is None:
            max_iter = self.params.max_generations

        self.initialize_population(dim, bounds)

        for i in range(self.params.population_size):
            self.fitness[i] = objective_func(self.population[i]) + 1000

        best_idx = np.argmax(self.fitness)
        self.gbest = self.population[best_idx].copy()
        self.gbest_fitness = self.fitness[best_idx]
        self.prey = self.gbest.copy()

        for gen in range(max_iter):
            self.generation = gen
            bes_pop = self.bes_exploration()
            go_pop = self.go_exploitation()

            bes_fitness = np.array([objective_func(x) + 1000 for x in bes_pop])
            go_fitness = np.array([objective_func(x) + 1000 for x in go_pop])

            improvement = max(np.max(bes_fitness), np.max(go_fitness)) - self.gbest_fitness

            if improvement < 0.01:
                self.stagnation_count += 1
                if self.stagnation_count > self.params.stagnation_threshold:
                    offspring = self.hybrid_crossover(bes_pop, go_pop)
                    offspring_fitness = np.array([objective_func(x) + 1000 for x in offspring])
                    worst_idx = np.argsort(self.fitness)[:len(offspring)]
                    self.population[worst_idx] = offspring
                    self.fitness[worst_idx] = offspring_fitness
                    self.stagnation_count = 0
            else:
                self.stagnation_count = 0
                if np.mean(bes_fitness) > np.mean(go_fitness):
                    self.population = bes_pop
                    self.fitness = bes_fitness
                else:
                    self.population = go_pop
                    self.fitness = go_fitness

            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.gbest_fitness:
                self.gbest = self.population[best_idx].copy()
                self.gbest_fitness = self.fitness[best_idx]
                self.prey = self.gbest.copy()

        return self.gbest, self.gbest_fitness
