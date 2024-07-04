import numpy as np

# Étape 1 - Initialisation

# Cette étape consiste à générer une population initiale de bernacles de manière aléatoire dans les limites spécifiées.
# Chaque individu de la population est évalué pour déterminer sa valeur de fitness.


class BarnaclesMatingOptimizer:
    def __init__(self, benchmark_function, population_size = 50, dimension = 30, max_iterations = 1000, pl = 0.5, bounds = (-10, 10)):
        self.benchmark_function = benchmark_function
        self.population_size = population_size
        self.dimension = dimension
        self.max_iterations = max_iterations
        self.pl = pl  # Mating probability
        self.bounds = bounds
        
        # Initialisation (Flowchart Step "Initialization")
        self.population = np.random.uniform(low = bounds[0], high = bounds[1], size = (population_size, dimension))
        self.fitness = np.apply_along_axis(benchmark_function, 1, self.population)
        self.best_solution = self.population[np.argmin(self.fitness)]
        self.best_fitness = min(self.fitness)
        self.fitness_history = [] 
        
# Étape 2 - Processus d'Évaluation

# Pendant le processus d'évaluation, la population actuelle est triée en fonction des valeurs de fitness.
# Les meilleures solutions trouvées jusqu'à présent sont conservées pour être utilisées dans les étapes suivantes.

    def evaluation_process(self):
        # Sort the population based on fitness (Flowchart Step "Evaluation process, sorting and store the best solutions so far")
        sorted_indices = np.argsort(self.fitness)
        sorted_population = self.population[sorted_indices]
        sorted_fitness = self.fitness[sorted_indices]
        return sorted_population, sorted_fitness

# Étape 3 - Sélection

# La sélection des parents pour générer de nouveaux descendants est effectuée en utilisant les équations (2) et (3) du document de référence.
# Ces équations déterminent comment les parents sont choisis en fonction de leurs valeurs de fitness.

    def mating_process(self):
        new_population = []
        for i in range(self.population_size // 2):
            # Sélection des parents (Flowchart Step "Selections using eqns. (2) & (3)")
            parents = self.population[np.random.choice(self.population_size, 2, replace = False)]
            k = np.abs(parents[0] - parents[1]).sum()  # Calcul de k
            
            if k <= self.pl:
                # Génération des descendants en utilisant l'équation (4)
                p = np.random.normal()
                q = 1 - p
                offspring1 = p * parents[0] + q * parents[1]
                offspring2 = p * parents[1] + q * parents[0]
            else:
                # Génération des descendants en utilisant l'équation (5)
                offspring1 = np.random.rand() * parents[1]
                offspring2 = np.random.rand() * parents[0]

# Étape 4 - Génération des Descendants

# La génération des descendants se fait selon deux stratégies, en fonction de la comparaison de 𝑘 avec 𝑝𝑙.
# Si 𝑘 est supérieur à 𝑝𝑙, les descendants sont générés en utilisant l'équation (5).
# Sinon, l'équation (4) est utilisée. Les nouveaux descendants sont ensuite vérifiés pour s'assurer qu'ils restent dans les limites spécifiées.

            # Check boundaries (Flowchart Step "Solutions out of bound? Pegging at boundaries")
            offspring1 = np.clip(offspring1, self.bounds[0], self.bounds[1])
            offspring2 = np.clip(offspring2, self.bounds[0], self.bounds[1])
            
            new_population.append(offspring1)
            new_population.append(offspring2)
        
        return np.array(new_population)

# Étape 5 - Tri et Conservation des Meilleures Solutions

# Après la génération des descendants, la population combinée (ancienne et nouvelle) est triée en fonction de la fitness,
# et les meilleurs individus sont conservés. Cette étape garantit que l'algorithme converge vers les solutions optimales.

    def optimize(self):
        for iteration in range(self.max_iterations):
            # Step 2: Evaluation process, sorting, and storing the best solutions so far (Flowchart Step "Evaluation process, sorting and store the best solutions so far")
            self.population, self.fitness = self.evaluation_process()
            # Generate new population through mating process
            new_population = self.mating_process()
            # Combine old and new populations
            self.population = np.vstack((self.population, new_population))
            # Evaluate the fitness of the combined population
            self.fitness = np.apply_along_axis(self.benchmark_function, 1, self.population)
            # Sort the population and select the top individuals
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices][:self.population_size]
            self.fitness = self.fitness[sorted_indices][:self.population_size]
            
            # Update the best solution found so far (Flowchart Step "Evaluation process, sorting and store the best solutions so far")
            current_best_fitness = min(self.fitness)
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[np.argmin(self.fitness)]
            
            self.fitness_history.append(self.best_fitness)
            # Print the best fitness at the current iteration
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, self.fitness_history
