import matplotlib.pyplot as plt
import numpy as np
from BMO import BarnaclesMatingOptimizer
from SMO import SMO
from IASO import ImprovedAtomSearchOptimization
from B_FUNCTIONS import *

bmo = BarnaclesMatingOptimizer(benchmark_function=sphere_function, population_size=50, dimension=30, max_iterations=100, bounds=(-100, 100))
_, _, bmo_fitness_history = bmo.optimize()

smo = SMO(100, 50, 30, -100, 100, sphere_function)
_, _, smo_fitness_history = smo.optimize()

iaso = ImprovedAtomSearchOptimization(sphere_function, num_atoms=20, num_iterations=100, dim=30, lb=-100, ub=100, sa_temp=10, sa_cooling_rate=0.99, sa_min_temp=2e-5)
_, _, iaso_fitness_history = iaso.optimize()

print(f"BarnaclesMatingOptimizer final fitness: {bmo_fitness_history[-1]}")
print(f"SMO final fitness: {smo_fitness_history[-1]}")
print(f"IASO final fitness: {iaso_fitness_history[-1]}")

plt.figure(figsize=(10, 6))
plt.plot(iaso_fitness_history, label='IASO')

plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Comparison of Optimization Algorithms')
plt.legend()
plt.grid(True)
plt.show()

final_fitness_values = [('BarnaclesMatingOptimizer', bmo_fitness_history[-1]),('SMO', smo_fitness_history[-1]),('IASO', iaso_fitness_history[-1]),]

print(f"{'Algorithm':<30} {'Final Fitness':<20}")
print("-" * 50)
for algo, fitness in final_fitness_values:
    print(f"{algo:<30} {fitness:<20.6f}")
