"""
This script performs hyperparameter optimization for a PyTorch-based MLP regression
using the Slime Mould Algorithm (SMA). The user selects an Excel dataset, and the
MLP hyperparameters (hidden layer size, max iterations, activation function) are
optimized to minimize MAE on the test set.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mealpy import IntegerVar, SMA, Problem
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open a file selection dialog to allow the user to choose an Excel file
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx;*.xls")])
    return file_path

# Read the Excel file
file_path = select_file()  # Select file
if file_path:
    # Read Excel file
    data = pd.read_excel(file_path)
    
    # Data preprocessing, first five columns as features, sixth column as target
    X = data.iloc[0:, 0:5].values  # Extract feature data
    y = data.iloc[0:, 5].values    # Extract target data

    # Set the search range for hyperparameters
    my_bounds = [
        IntegerVar(lb=10, ub=200, name="hidden_layer_sizes_paras"),  # Hidden layer size
        IntegerVar(lb=20, ub=5000, name="max_iter_paras"),           # Maximum iterations
        IntegerVar(lb=0, ub=3, name="activation_paras")              # Activation function selection (0: ReLU, 1: Sigmoid, 2: Tanh, 3: Identity)
    ]

    # Define optimization problem
    class MlpOptimizedProblem(Problem):
        def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
            self.data = data
            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, x):
            x_decoded = self.decode_solution(x)
            hidden_layer_sizes_paras = int(x_decoded["hidden_layer_sizes_paras"])
            max_iter_paras = int(x_decoded["max_iter_paras"])
            activation_paras = int(x_decoded["activation_paras"])

            # Map activation_paras to corresponding activation function
            activation_map = {0: 'relu', 1: 'logistic', 2: 'tanh', 3: 'identity'}
            activation = activation_map[activation_paras]

            # Move data to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_train_tensor = torch.tensor(self.data["X_train"], dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(self.data["y_train"], dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(self.data["X_test"], dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(self.data["y_test"], dtype=torch.float32).to(device)

            # Initialize MLP regression model
            mlp = nn.Sequential(
                nn.Linear(X_train_tensor.shape[1], hidden_layer_sizes_paras),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),  # Example of ReLU or Tanh
                nn.Linear(hidden_layer_sizes_paras, 1)
            ).to(device)

            optimizer = optim.Adam(mlp.parameters())
            criterion = nn.MSELoss()

            # Fit the model
            mlp.train()
            for epoch in range(max_iter_paras):
                optimizer.zero_grad()
                output = mlp(X_train_tensor)
                loss = criterion(output.view(-1), y_train_tensor)
                loss.backward()
                optimizer.step()

            # Prediction
            mlp.eval()
            with torch.no_grad():
                y_predict = mlp(X_test_tensor).cpu().numpy()
            mae = metrics.mean_absolute_error(y_test_tensor.cpu().numpy(), y_predict)
            return mae  # Objective function is MAE, minimize MAE

    # Split data into training and testing sets, ratio 7:3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    # Initialize problem instance
    problem = MlpOptimizedProblem(bounds=my_bounds, minmax="min", data=data)

    # Initialize SMA algorithm and perform optimization
    model = SMA.OriginalSMA(epoch=100, pop_size=50)
    model.solve(problem)

    # Output optimization results
    print(f"Best agent: {model.g_best}")
    print(f"Best solution: {model.g_best.solution}")
    print(f"Best MAE: {model.g_best.target.fitness}")
    print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")

else:
    print("No file selected")
