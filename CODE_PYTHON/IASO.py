import numpy as np

# Atom Search Optimization Algorithm
class AtomSearchOptimization:
    def __init__(self, func, num_atoms=30, num_iterations=100, dim=30, lb=-100, ub=100):
        self.func = func
        self.num_atoms = num_atoms
        self.num_iterations = num_iterations
        self.dim = dim
        self.lb = lb
        self.ub = ub
        # Randomly initialize positions and velocities
        self.positions = np.random.uniform(lb, ub, (num_atoms, dim))
        self.velocities = np.random.uniform(-1, 1, (num_atoms, dim))
        self.best_position = None
        self.best_fitness = np.inf
        self.fitness_history = []

    def update_alpha_beta(self, t):
        T = self.num_iterations
        # Update alpha and beta values based on the current iteration
        self.alpha = 0.1 * (np.sin(np.pi * t / (2 * T)) + 1)
        self.beta = 0.1 * (np.cos(np.pi * t / T) + 1)

    def optimize(self):
        for t in range(self.num_iterations):
            self.update_alpha_beta(t)  # Update alpha and beta values

            # Calculate the fitness value for each atom
            fitness = np.array([self.func(atom) for atom in self.positions])
            self.fitness_history.append(np.min(fitness))
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_position = self.positions[min_fitness_idx].copy()

            # Print the current iteration and best fitness
            print(f"ASO Iteration: {t}, Best Fitness: {self.best_fitness}")

            # Calculate mass based on fitness values
            worst_fitness = np.max(fitness)
            best_fitness = np.min(fitness)
            if worst_fitness == best_fitness:
                worst_fitness += 1e-10
            M = (fitness - worst_fitness) / (best_fitness - worst_fitness + 1e-10)
            mass = np.exp(-M)
            mass /= np.sum(mass)

            # Calculate K(t) based on the current iteration
            K = int(self.num_atoms - (self.num_atoms - 2) * np.sqrt(t / self.num_iterations))
            K_best = np.argsort(fitness)[:K]

            # Calculate interaction and constraint forces
            forces = np.zeros((self.num_atoms, self.dim))
            for i in range(self.num_atoms):
                for j in K_best:
                    if i != j:
                        r_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                        if r_ij == 0:
                            r_ij += 1e-10
                        F_ij = self.alpha * (self.positions[j] - self.positions[i]) / (r_ij ** 2)
                        forces[i] += np.random.rand() * F_ij

            lambda_t = self.beta * np.exp(-20 * t / self.num_iterations)
            constraint_forces = lambda_t * (self.best_position - self.positions)

            accelerations = (forces + constraint_forces) / mass[:, np.newaxis]

            self.velocities = np.random.uniform(-1, 1, (self.num_atoms, self.dim)) * self.velocities + accelerations
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lb, self.ub)

        return self.best_position, self.best_fitness, self.fitness_history

# Simulated Annealing Algorithm
class SimulatedAnnealing:
    def __init__(self, func, initial_solution, initial_temp=100, cooling_rate=0.99, min_temp=1e-3):
        self.func = func
        self.current_solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.current_fitness = func(initial_solution)
        self.best_fitness = self.current_fitness
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.fitness_history = []

    def optimize(self):
        iteration = 0
        while self.temperature > self.min_temp:
            iteration += 1
            # Generate new solution in the neighborhood
            new_solution = self.current_solution + np.random.uniform(-1, 1, len(self.current_solution))
            new_solution = np.clip(new_solution, -100, 100)
            new_fitness = self.func(new_solution)

            # Accept or reject the new solution based on SA acceptance criteria
            if new_fitness < self.current_fitness or np.random.rand() < np.exp((self.current_fitness - new_fitness) / self.temperature):
                self.current_solution = new_solution
                self.current_fitness = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = new_fitness

            # Store the best fitness in the fitness history
            self.fitness_history.append(self.best_fitness)
            print(f"SA Iteration: {iteration}, Temperature: {self.temperature:.2f}, Best Fitness: {self.best_fitness}")

            # Apply cooling schedule
            self.temperature *= self.cooling_rate

        return self.best_solution, self.best_fitness, self.fitness_history

# Improved Atom Search Optimization Algorithm
class ImprovedAtomSearchOptimization:
    def __init__(self, func, num_atoms=30, num_iterations=100, dim=30, lb=-100, ub=100, sa_temp=100, sa_cooling_rate=0.9, sa_min_temp=1e-3):
        self.func = func
        self.num_atoms = num_atoms
        self.num_iterations = num_iterations
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.sa_temp = sa_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_min_temp = sa_min_temp

    def optimize(self):
        # Perform ASO
        aso = AtomSearchOptimization(self.func, self.num_atoms, self.num_iterations, self.dim, self.lb, self.ub)
        best_position, best_fitness, aso_fitness_history = aso.optimize()

        # Perform SA using the best solution found by ASO
        sa = SimulatedAnnealing(self.func, best_position, self.sa_temp, self.sa_cooling_rate, self.sa_min_temp)
        best_position, best_fitness, sa_fitness_history = sa.optimize()

        # Concatenate fitness history from ASO and SA for complete history
        fitness_history = aso_fitness_history + sa_fitness_history
        return best_position, best_fitness, fitness_history
