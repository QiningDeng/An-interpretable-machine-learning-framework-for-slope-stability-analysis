"""
This script performs Decision Tree Regression (DTR) hyperparameter optimization
using the Slime Mould Algorithm (SMA). It allows the user to select an Excel
dataset, splits the data into training and test sets, defines the optimization
problem, and finds the best hyperparameters that minimize the Mean Absolute Error (MAE).
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mealpy import FloatVar, IntegerVar, SMA, Problem
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ===================== File Selection =====================
def select_file():
    """Open a file dialog to allow the user to select an Excel file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx;*.xls")])
    root.destroy()
    return file_path

# ===================== Load and Preprocess Data =====================
file_path = select_file()  # User selects the Excel file
if file_path:
    # Read Excel file
    data = pd.read_excel(file_path)
    
    # Data preprocessing: assume first 5 columns are features and the 6th column is the target
    X = data.iloc[:, 0:5].values  # Extract feature data
    y = data.iloc[:, 5].values    # Extract target data
    
    # Split the dataset into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Store split data in a dictionary for easy access
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    # ===================== Define Optimization Problem =====================
    class DtrOptimizedProblem(Problem):
        """Decision Tree Regression hyperparameter optimization problem"""
        def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
            self.data = data
            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, x):
            """Objective function: minimize Mean Absolute Error (MAE) on test set"""
            x_decoded = self.decode_solution(x)
            max_depth_paras = x_decoded["max_depth_paras"]
            min_samples_split_paras = x_decoded["min_samples_split_paras"]
            max_features_paras = x_decoded["max_features_paras"]

            # Initialize Decision Tree Regressor with hyperparameters
            dtr = DecisionTreeRegressor(
                max_depth=max_depth_paras,
                min_samples_split=min_samples_split_paras,
                max_features=max_features_paras,
                random_state=1
            )
            # Fit the model
            dtr.fit(self.data["X_train"], self.data["y_train"])
            # Predict on test set
            y_predict = dtr.predict(self.data["X_test"])
            # Compute Mean Absolute Error (MAE)
            mae = metrics.mean_absolute_error(self.data["y_test"], y_predict)
            return mae  # Minimize MAE

    # ===================== Define Hyperparameter Search Space =====================
    my_bounds = [
        IntegerVar(lb=1, ub=40, name="max_depth_paras"),        # max_depth range
        IntegerVar(lb=2, ub=20, name="min_samples_split_paras"),# min_samples_split range
        FloatVar(lb=0.1, ub=1, name="max_features_paras")       # max_features range
    ]

    # Initialize problem instance
    problem = DtrOptimizedProblem(bounds=my_bounds, minmax="min", data=data)

    # ===================== Run SMA Optimization =====================
    model = SMA.OriginalSMA(epoch=100, pop_size=50)
    model.solve(problem)

    # ===================== Output Results =====================
    print(f"Best agent: {model.g_best}")
    print(f"Best solution: {model.g_best.solution}")
    print(f"Best MAE: {model.g_best.target.fitness}")
    print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")

else:
    print("No file selected")
