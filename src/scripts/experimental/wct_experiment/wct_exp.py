
import random
import time
import numpy as np
from src.scripts.experimental.wrappers.experiment_model import BaseModel


class WallClockTimeExperiment():
    def __init__(self, model: BaseModel, n_samples: int=10000, iterations: int=5, seed: int=42):
        self.model = model
        self.n_samples = n_samples
        self.iterations = iterations
        self.seed = seed
        self.rng = random.Random(seed)
    
    def run(self, data):
        mean_times = np.zeros((self.iterations))
        
        samples = self.rng.choices(data, k=self.n_samples)
        
        for i in range(self.iterations):
            start_time = time.time()
            for sample in samples:
                _ = self.model(sample)    
            end_time = time.time()
            mean_times[i] = (end_time - start_time) / self.n_samples
        
        print(f'After {self.iterations} iterations the mean time for each iteration was:')
        for i in range(self.iterations):
            print(f'\t{i+1}: {mean_times[i]} per sample.')
        print(f'Mean time across all iterations is {np.mean(mean_times)} per sample.')