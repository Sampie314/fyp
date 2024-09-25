import random
import logging
import sys
import numpy as np
from typing import List, Dict, Tuple, Callable, Any
import torch.multiprocessing as mp
from functools import partial
import math
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from . import Utils

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

def fitness(train_func: Callable, chromosome: Dict[str, Any], X: np.array, y: np.array, t_X: np.array, t_y: np.array, crisp_t_y: np.array, cls, pred_col) -> float:
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
        X, y, t_X, t_y, crisp_t_y, cls, pred_col,
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

# def genetic_algorithm(train_func: Callable, X: np.array, y: np.array, t_X: np.array, t_y: np.array, crisp_t_y: Any, 
#                       initial_population_size: int = 50, final_population_size: int = 5, generations: int = 10, elite_size: int = 2) -> Tuple[Dict[str, Any], float]:
#     """
#     Perform the genetic algorithm to find the best hyperparameters.

#     Args:
#         train_func (Callable): The training function to evaluate chromosomes.
#         X, y, t_X, t_y, crisp_t_y: Data for training and evaluation.
#         population_size (int, optional): The size of the population. Defaults to 20.
#         generations (int, optional): The number of generations to evolve. Defaults to 10.
#         elite_size (int, optional): The number of top performers to keep in each generation. Defaults to 2.

#     Returns:
#         Tuple[Dict[str, Any], float]: The best chromosome and its fitness score.
#     """
#     def calculate_population_size(generation: int) -> int:
#         """Calculate the population size for a given generation."""
#         return int(initial_population_size - (initial_population_size - final_population_size) * (generation / (generations - 1)))
        
#     population = generate_population(initial_population_size)
    
#     for generation in range(generations):
#         logger.info(f"Generation {generation + 1}/{generations}")
        
#         # Evaluate fitness
#         fitness_scores = [(chromosome, fitness(train_func, chromosome, X, y, t_X, t_y, crisp_t_y)) for chromosome in population]
#         fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
#         # Select elite
#         new_population = [chromosome for chromosome, _ in fitness_scores[:elite_size]]
        
#         # Generate offspring
#         population_size = calculate_population_size(generation)
#         while len(new_population) < population_size:
#             parent1, parent2 = random.sample(population, 2)
#             child = crossover(parent1, parent2)
#             child = mutation(child)
#             new_population.append(child)
        
#         population = new_population
        
#         best_chromosome, best_fitness = fitness_scores[0]
#         logger.info(f"Best fitness: {best_fitness}")
#         logger.info(f"Best chromosome: {best_chromosome}")
    
#     return fitness_scores[0]

def calculate_population_size(generation: int) -> int:
    """Calculate the population size for a given generation."""
    return int(initial_population_size - (initial_population_size - final_population_size) * (generation / (generations - 1)))

def evaluate_fitness(chromosome, train_func, X, y, t_X, t_y, crisp_t_y, cls, pred_col):
    """Evaluate the fitness of a single chromosome."""
    return chromosome, fitness(train_func, chromosome, X, y, t_X, t_y, crisp_t_y, cls, pred_col)

# Convert inputs to PyTorch tensors if they're NumPy arrays
def to_tensor(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    
def genetic_algorithm(train_func: Callable, X: torch.Tensor, y: torch.Tensor, t_X: torch.Tensor, t_y: torch.Tensor, crisp_t_y: torch.Tensor, cls, pred_col,
                      initial_population_size: int = 50, final_population_size: int = 5, generations: int = 10, elite_size: int = 2) -> Tuple[Dict[str, Any], float]:
    """
    Perform the genetic algorithm to find the best hyperparameters with parallel chromosome evaluation.

    Args:
        train_func (Callable): The training function to evaluate chromosomes.
        X, y, t_X, t_y, crisp_t_y: PyTorch tensors for training and evaluation.
        initial_population_size (int): The initial size of the population.
        final_population_size (int): The final size of the population.
        generations (int): The number of generations to evolve.
        elite_size (int): The number of top performers to keep in each generation.

    Returns:
        Tuple[Dict[str, Any], float]: The best chromosome and its fitness score.
    """
    # X = to_tensor(X)
    # y = to_tensor(y)
    # t_X = to_tensor(t_X)
    # t_y = to_tensor(t_y)
    # crisp_t_y = to_tensor(crisp_t_y)

    # Ensure all tensors are on CPU for multiprocessing
    # X, y, t_X, t_y, crisp_t_y = X.cpu(), y.cpu(), t_X.cpu(), t_y.cpu(), crisp_t_y.cpu()
    
    population = generate_population(initial_population_size)
    
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}/{generations}")
        
        # Evaluate fitness in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            fitness_scores = pool.map(partial(evaluate_fitness, train_func=train_func, X=X, y=y, t_X=t_X, t_y=t_y, crisp_t_y=crisp_t_y, cls=cls, pred_col=pred_col), population)
        
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

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=10, ff_dim=32, num_transformer_blocks=4, mlp_units=256, dropout=0.25, noHiddenLayers=1, sigmoidOrSoftmax=0):
        super(TransformerModel, self).__init__()

        # print(f"Initializing TransformerModel with input_dim: {input_dim}, output_dim: {output_dim}")

        # # Ensure input_dim is divisible by num_heads
        # if input_dim % num_heads != 0:
        #     new_input_dim = (input_dim // num_heads + 1) * num_heads
        #     print(f"Adjusting input_dim from {input_dim} to {new_input_dim} to be divisible by num_heads ({num_heads})")
        #     input_dim = new_input_dim

        # Encoder layer with ff_dim and dropout
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_transformer_blocks)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # MLP layers
        fcList = [torch.nn.Linear(input_dim, mlp_units), torch.nn.ReLU(), self.dropout]
        for i in range(noHiddenLayers):
            fcList.extend([
                torch.nn.Linear(mlp_units, mlp_units//2),
                torch.nn.ReLU(),
                self.dropout
            ])
            mlp_units = mlp_units//2
        fcList.append(torch.nn.Linear(mlp_units, output_dim))
        
        # Output activation
        if sigmoidOrSoftmax == 0:
            fcList.append(torch.nn.Sigmoid())
        else:
            fcList.append(torch.nn.Softmax(dim=-1))
        
        self.fc = torch.nn.Sequential(*fcList)

    def forward(self, src):
        # src shape: (batch_size, seq_length, input_dim)
        
        # Pass input through the transformer encoder
        encoder_output = self.transformer_encoder(src)
        
        # Apply dropout after the transformer encoder
        encoder_output = self.dropout(encoder_output)
        
        # Pass through the MLP layers
        output = self.fc(encoder_output)
        return output
    
def increaseInstancesExtreme(train, thresholdToIncrease=0.03):
    extraData = train[(train[f"Close_t+{yTarget}"]>thresholdToIncrease) | (train[f"Close_t+{yTarget}"]<-thresholdToIncrease)]
    return pd.concat([train, extraData] ,axis = 0)

def train_model(X, Y, X_test, Y_test, # fuzzified inputs
                   Y_test_raw, # crisp value
                cls, pred_col,
                   num_heads, feed_forward_dim, num_transformer_blocks, mlp_units, dropout_rate, 
                   learning_rate, num_mlp_layers, num_epochs, activation_function, batch_size):

    OUTPUT_FREQ = 50
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load and pad data
    x_padded, x_test_padded = Utils.padData(X, X_test, math.ceil(X.shape[1] / num_heads) * num_heads - X.shape[1])

    # Initialize the model
    model = TransformerModel(
        input_dim=x_padded.shape[1], 
        output_dim=Y.shape[1],
        num_heads=num_heads, 
        ff_dim=feed_forward_dim, 
        num_transformer_blocks=num_transformer_blocks, 
        mlp_units=mlp_units, 
        dropout=dropout_rate, 
        sigmoidOrSoftmax=activation_function
    ).double()

    # Send model to the detected device
    model = model.to(device)

    # Set loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for training
    assert x_padded.shape[0] == Y.shape[0]
    train_dataset = TensorDataset(torch.from_numpy(x_padded), torch.from_numpy(Y))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        all_preds = []
        all_targets = []

        for inputs, targets in train_dataloader:
            # Move data to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Collect predictions and targets for R² score computation
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        # Concatenate all batch predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if epoch % OUTPUT_FREQ == 0:
            # Compute R² score
            r2 = r2_score(all_targets, all_preds)
            log_return_r2 = eval_model(model, x_test_padded, Y_test, Y_test_raw, cls, pred_col)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss / len(train_dataloader):.4f} | Train Cluster R² Score: {r2:.4f} | Test Log Return R² Score: {log_return_r2:.4f}')

    # return all_preds, all_targets

    # Evaluate on test data
    test_dataset = TensorDataset(torch.from_numpy(x_test_padded), torch.from_numpy(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=512*4, shuffle=False)

    model.eval()
    pred_list = []
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            pred = model(inputs.to(device)).cpu()
            pred_list.append(pred)

    mse = Utils.testData(np.concatenate(pred_list), Y_test)
    # print(f'Test Data MSE: {mse:.7f}')

    # Calculate R2 score
    pred = np.concatenate(pred_list)
    pred[np.isnan(pred)] = 0

    pred_log_returns = cls.deFuzzify(pred, pred_col)
    pred_log_returns = np.nan_to_num(pred_log_returns, nan=0, posinf=0, neginf=0)
    
    # pred_closing = addResToClosing(cls, res)[:-yTarget]
    # actual_closing = cls.test.Close[yTarget:].to_numpy()
    # r2_score_value = r2_score(actual_closing, pred_closing)
    
    r2_score_value = r2_score(Y_test_raw, pred_log_returns)
    print("Test R2 Score:", r2_score_value)

    return model, r2_score_value, pred

def eval_model(model, X: np.array, Y: np.array, Y_crisp: np.array, cls, pred_col):
    """
    X & Y are fuzzified inputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device}")    

    model = model.to(device)
    model.eval()
    pred_list = []

    test_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    test_dataloader = DataLoader(test_dataset, batch_size=512*4, shuffle=False)
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            pred = model(inputs.to(device)).cpu()
            pred_list.append(pred)

    # mse = Utils.testData(np.concatenate(pred_list), Y_test)
    # print(f'Test Data MSE: {mse:.7f}')

    # Calculate R2 score
    pred = np.concatenate(pred_list)
    pred[np.isnan(pred)] = 0

    pred_log_returns = cls.deFuzzify(pred, pred_col)
    pred_log_returns = np.nan_to_num(pred_log_returns, nan=0, posinf=0, neginf=0)
    
    r2_score_value = Utils.custom_r2_score(Y_crisp, pred_log_returns)
    # print("Test R2 Score:", r2_score_value)    
    return r2_score_value


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