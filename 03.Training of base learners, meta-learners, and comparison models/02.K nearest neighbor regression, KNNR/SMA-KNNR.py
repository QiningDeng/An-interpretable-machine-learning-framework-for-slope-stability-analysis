"""
This script performs K-Nearest Neighbors Regression (KNN) hyperparameter optimization
using the Slime Mould Algorithm (SMA). It allows the user to select an Excel dataset,
splits the data into training and test sets (70%/30%), defines the SMA optimization
problem to minimize Mean Absolute Error (MAE), and outputs the best solution and parameters.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mealpy import IntegerVar, SMA, Problem
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ===================== 1. File Selection =====================
def select_file():
    """Open a file dialog for selecting an Excel file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx;*.xls")])
    root.destroy()
    return file_path

# ===================== 2. Load Dataset =====================
file_path = select_file()  # Prompt user to select file
if file_path:
    data = pd.read_excel(file_path)
    
    # Data preprocessing: first 5 columns as features, 6th column as target
    X = data.iloc[:, 0:5].values  # Extract feature data
    y = data.iloc[:, 5].values    # Extract target data

    # ===================== 3. Hyperparameter Search Space =====================
    my_bounds = [
        IntegerVar(lb=1, ub=20, name="n_neighbors_paras"),  # n_neighbors
        IntegerVar(lb=0, ub=1, name="weights_paras"),       # weights (0='uniform', 1='distance')
        IntegerVar(lb=1, ub=2, name="p_paras")              # p (1=Manhattan, 2=Euclidean)
    ]

    # ===================== 4. Define Optimization Problem =====================
    class KnnOptimizedProblem(Problem):
        """Define SMA optimization problem for KNN hyperparameters"""
        def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
            self.data = data
            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, x):
            """Objective function: decode solution, train KNN, return MAE"""
            x_decoded = self.decode_solution(x)
            n_neighbors_paras, weights_paras, p_paras = (
                x_decoded["n_neighbors_paras"], 
                x_decoded["weights_paras"], 
                x_decoded["p_paras"]
            )

            # Map integer weight parameter to string value
            weights_map = {0: 'uniform', 1: 'distance'}
            weights_paras = weights_map[weights_paras]

            # Initialize KNN regressor
            knn = KNeighborsRegressor(n_neighbors=n_neighbors_paras, weights=weights_paras, p=p_paras)
            # Fit the model on training data
            knn.fit(self.data["X_train"], self.data["y_train"])
            # Predict on test set
            y_predict = knn.predict(self.data["X_test"])
            # Compute Mean Absolute Error (MAE)
            mae = metrics.mean_absolute_error(self.data["y_test"], y_predict)
            return mae  # Objective: minimize MAE

    # ===================== 5. Split Data =====================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    # ===================== 6. Initialize Problem and SMA Algorithm =====================
    problem = KnnOptimizedProblem(bounds=my_bounds, minmax="min", data=data)

    # Initialize SMA algorithm and solve the optimization problem
    model = SMA.OriginalSMA(epoch=100, pop_size=50)
    model.solve(problem)

    # ===================== 7. Output Best Results =====================
    print(f"Best agent: {model.g_best}")
    print(f"Best solution: {model.g_best.solution}")
    print(f"Best MAE: {model.g_best.target.fitness}")
    print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")

else:
    print("No file selected")
