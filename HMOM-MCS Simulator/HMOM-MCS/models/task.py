# ================== FILE: models/task.py ==================

import numpy as np

class Task:
    """Task entity with all attributes from Table 3"""
    def __init__(self, task_id: int, params):
        self.id = task_id
        self.location = np.random.uniform(0, params.grid_size[0], 2)
        self.qoi_requirement = np.random.uniform(0.5, 1.0)
        self.budget = np.random.uniform(10, 50) * 10
        self.deadline = np.random.uniform(10, 30)
        self.freshness_threshold = np.random.uniform(1, 3)
        self.participation_quota = np.random.randint(5, 21)
        self.assigned_users = []
        self.aoi = 0.0
        self.coverage = 0.0
        self.completed = False
        self.value = self.budget * self.qoi_requirement * 1.2
