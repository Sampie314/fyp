import random
import logging
import sys
import numpy as np
from typing import List, Dict, Tuple, Callable, Any

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger with the given name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger object.
    """
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

def hyperparameters(parameters=None) -> Dict[str, List[Any]]:
    """
    Define the hyperparameters and their possible values for the genetic algorithm.

    Returns:
        Dict[str, List[Any]]: A dictionary of hyperparameters and their possible values.
    """
    if parameters is None:
        parameters = {
            'num_heads': [1, 2, 3, 4],
            'feed_forward_dim': [32, 64, 128],
            'num_transformer_blocks': [1, 2, 3],
            'mlp_units': [64, 128, 256, 512],
            'dropout_rate': [0.01, 0.05, 0.1, 0.2],
            'learning_rate': [0.001, 0.0001, 0.00001, 0.000005],
            'num_mlp_layers': [3, 5, 8],
            'num_epochs': [25, 50, 100, 200, 500],
            'activation_function': [0, 1],  # 0 for sigmoid, 1 for softmax
            'batch_size': [128, 256, 512, 1024, 2048]
        }
    return parameters

def generate_population(size: int) -> List[Dict[str, Any]]:
    """
    Generate a population of unique chromosomes.

    Args:
        size (int): The desired size of the population.

    Returns:
        List[Dict[str, Any]]: A list of chromosomes, where each chromosome is a dictionary
        of hyperparameter names and their randomly chosen values.
    """
    parameters = hyperparameters()
    population = []
    
    while len(population) < size:
        chromosome = {key: random.choice(value) for key, value in parameters.items()}
        if chromosome not in population:
            population.append(chromosome)
    return population

def crossover(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform crossover between two parent chromosomes to create a child chromosome.

    Args:
        parent1 (Dict[str, Any]): The first parent chromosome.
        parent2 (Dict[str, Any]): The second parent chromosome.

    Returns:
        Dict[str, Any]: A new child chromosome created from the parents.
    """
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutation(chromosome: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
    """
    Apply mutation to a chromosome based on the given mutation rate.

    Args:
        chromosome (Dict[str, Any]): The chromosome to mutate.
        mutation_rate (float, optional): The probability of mutating each gene. Defaults to 0.1.

    Returns:
        Dict[str, Any]: The mutated chromosome.
    """
    parameters = hyperparameters()
    for key in chromosome.keys():
        if random.random() < mutation_rate:
            chromosome[key] = random.choice(parameters[key])
    return chromosome

def fitness(train_func: Callable, chromosome: Dict[str, Any], X: np.array, y: np.array, t_X: np.array, t_y: np.array, crisp_t_y: np.array) -> float:
    """
    Calculate the fitness of a chromosome using the provided training function.

    Args:
        train_func (Callable): The training function to evaluate the chromosome.
        chromosome (Dict[str, Any]): The chromosome to evaluate.
        X, y, t_X, t_y, crisp_t_y: Data for training and evaluation.

    Returns:
        float: The fitness score (R2 score) of the chromosome.
    """
    model, r2, pred = train_func(
        X, y, t_X, t_y, crisp_t_y,
        num_heads=chromosome['num_heads'],
        feed_forward_dim=chromosome['feed_forward_dim'],
        num_transformer_blocks=chromosome['num_transformer_blocks'],
        mlp_units=chromosome['mlp_units'],
        dropout_rate=chromosome['dropout_rate'],
        learning_rate=chromosome['learning_rate'],
        num_mlp_layers=chromosome['num_mlp_layers'],
        num_epochs=chromosome['num_epochs'],
        activation_function=chromosome['activation_function'],
        batch_size=chromosome['batch_size']
    )
    return r2

def genetic_algorithm(train_func: Callable, X: np.array, y: np.array, t_X: np.array, t_y: np.array, crisp_t_y: Any, 
                      population_size: int = 50, final_population_size: int = 5, generations: int = 10, elite_size: int = 2) -> Tuple[Dict[str, Any], float]:
    """
    Perform the genetic algorithm to find the best hyperparameters.

    Args:
        train_func (Callable): The training function to evaluate chromosomes.
        X, y, t_X, t_y, crisp_t_y: Data for training and evaluation.
        population_size (int, optional): The size of the population. Defaults to 20.
        generations (int, optional): The number of generations to evolve. Defaults to 10.
        elite_size (int, optional): The number of top performers to keep in each generation. Defaults to 2.

    Returns:
        Tuple[Dict[str, Any], float]: The best chromosome and its fitness score.
    """
    def calculate_population_size(generation: int) -> int:
        """Calculate the population size for a given generation."""
        return int(initial_population_size - (initial_population_size - final_population_size) * (generation / (generations - 1)))
        
    population = generate_population(population_size)
    
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness
        fitness_scores = [(chromosome, fitness(train_func, chromosome, X, y, t_X, t_y, crisp_t_y)) for chromosome in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select elite
        new_population = [chromosome for chromosome, _ in fitness_scores[:elite_size]]
        
        # Generate offspring
        population_size = calculate_population_size(generation)
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
    # Usage example - this doesn't work as is, you need to replace the train_func with your own function in notebook
    best_chromosome, best_fitness = genetic_algorithm(X, y, t_X, t_y, crisp_t_y)
    print("Best hyperparameters:", best_chromosome)
    print("Best R2 score:", best_fitness)

    # Train final model with best hyperparameters
    final_model, final_r2, final_pred = train_func(
        X, y, t_X, t_y, crisp_t_y,
        **best_chromosome
    )
    print("Final model R2 score:", final_r2)