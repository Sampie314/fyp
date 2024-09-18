import random
import logging
import sys

# Configure logging
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicate logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger(__name__)

def hyperparameters():
    parameters = {
        'num_heads': [1, 2, 3, 4],
        'feed_forward_dim': [32, 64, 128],
        'num_transformer_blocks': [1, 2, 3],
        'mlp_units': [64, 128, 256, 512],
        'dropout_rate': [0.05, 0.1, 0.2],
        'learning_rate': [0.00005, 0.00001, 0.000005],
        'num_mlp_layers': [3, 5, 8],
        'num_epochs': [25, 50, 100, 500],
        'activation_function': [0, 1]  # 0 for sigmoid, 1 for softmax
    }
    return parameters

def generate_population(size: int):
    parameters = hyperparameters()
    population = []
    
    while len(population) < size:
        chromosome = {key: random.choice(value) for key, value in parameters.items()}
        if chromosome not in population:
            population.append(chromosome)
    return population

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutation(chromosome, mutation_rate=0.1):
    parameters = hyperparameters()
    for key in chromosome.keys():
        if random.random() < mutation_rate:
            chromosome[key] = random.choice(parameters[key])
    return chromosome

def fitness(train_func, chromosome, X, y, t_X, t_y, crisp_t_y):
    _, r2, _ = train_func(
        X, y, t_X, t_y, crisp_t_y,
        num_heads=chromosome['num_heads'],
        feed_forward_dim=chromosome['feed_forward_dim'],
        num_transformer_blocks=chromosome['num_transformer_blocks'],
        mlp_units=chromosome['mlp_units'],
        dropout_rate=chromosome['dropout_rate'],
        learning_rate=chromosome['learning_rate'],
        num_mlp_layers=chromosome['num_mlp_layers'],
        num_epochs=chromosome['num_epochs'],
        activation_function=chromosome['activation_function']
    )
    return r2

def genetic_algorithm(train_func, X, y, t_X, t_y, crisp_t_y, population_size=20, generations=10, elite_size=2):
    population = generate_population(population_size)
    
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness
        fitness_scores = [(chromosome, fitness(train_func, chromosome, X, y, t_X, t_y, crisp_t_y)) for chromosome in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select elite
        new_population = [chromosome for chromosome, _ in fitness_scores[:elite_size]]
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        
        population = new_population
        
        best_chromosome, best_fitness = fitness_scores[0]
        logger.info(f"Best fitness: {best_fitness}")
        logger.info(f"Best chromosome: {best_chromosome}")
    
    return fitness_scores[0]



if __name__ == "__main__":
    # Usage - this doesn't work as is, you need to replace the train_func with your own function in notebook
    best_chromosome, best_fitness = genetic_algorithm(X, y, t_X, t_y, crisp_t_y)
    print("Best hyperparameters:", best_chromosome)
    print("Best R2 score:", best_fitness)

    # Train final model with best hyperparameters
    final_model, final_r2, final_pred = train_func(
        X, y, t_X, t_y, crisp_t_y,
        **best_chromosome
    )
    print("Final model R2 score:", final_r2)