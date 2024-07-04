import numpy as np

class SMO:
    # Étape 1 - Initialisation
    def __init__(self, num_iterations, num_followers, dim, lb, ub, fitness_function):
        self.num_iterations = num_iterations
        self.num_followers = num_followers
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.fitness_function = fitness_function
        # Génération des followers de manière aléatoire dans les limites spécifiées
        self.followers = np.random.uniform(low = lb, high = ub, size = (num_followers, dim))
        # Création de la matrice sociale
        self.social_matrix = np.hstack((self.followers, np.zeros((num_followers, 1))))
        self.leader = None
        self.fitness_history = []

    # Étape 2 - Initialisation de la matrice sociale
    def initialize_social_matrix(self):
        for i in range(self.num_followers):
            # Calcul du fitness pour chaque follower
            self.social_matrix[i, -1] = self.fitness_function(self.social_matrix[i, :-1])
        # Détermination du leader initial (meilleur fitness)
        self.leader = self.social_matrix[np.argmin(self.social_matrix[:, -1]), :]

    # Étape 3 - Mise à jour des followers
    def update_followers(self):
        for i in range(self.num_followers):
            fitness_i = self.social_matrix[i, -1]
            # Calcul de la différence par rapport au leader
            difference = (self.leader[-1] - fitness_i) / fitness_i if fitness_i != 0 else np.random.uniform(0, 1)
            if difference == 0:
                difference = np.random.uniform(0, 1)
            # Mise à jour de la position des followers
            self.social_matrix[i, :-1] += difference * self.social_matrix[i, :-1]
            # Vérification des limites et ajustement
            self.social_matrix[i, :-1] = np.clip(self.social_matrix[i, :-1], self.lb, self.ub)
            # Calcul du nouveau fitness
            new_fitness = self.fitness_function(self.social_matrix[i, :-1])
            if new_fitness < self.social_matrix[i, -1]:
                self.social_matrix[i, -1] = new_fitness

    # Étape 4 - Boucle d'optimisation
    def optimize(self):
        self.initialize_social_matrix()
        for iteration in range(self.num_iterations):
            self.update_followers()
            # Mise à jour du leader
            self.leader = self.social_matrix[np.argmin(self.social_matrix[:, -1]), :]
            best_fitness = self.leader[-1]
            self.fitness_history.append(best_fitness)
            print(f"Iteration: {iteration + 1}, Best Fitness: {best_fitness}")
        # Retourne le meilleur solution et son fitness
        return self.leader[:-1], self.leader[-1], self.fitness_history