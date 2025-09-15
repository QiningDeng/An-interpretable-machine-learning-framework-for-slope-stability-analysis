"""
This script uses the SMA (Slime Mould Algorithm) metaheuristic optimizer 
to search for the best hyperparameters of an ElasticNet regression model. 
It allows the user to select an Excel file, extracts features and target data, 
defines an optimization problem, sets hyperparameter bounds, and solves the 
problem to minimize Mean Absolute Error (MAE).
"""

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mealpy import FloatVar, IntegerVar, SMA, Problem
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ========== 1. File selection ==========
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(title="Select Excel file", filetypes=[("Excel Files", "*.xlsx;*.xls")])
    return file_path

file_path = select_file()
if not file_path:
    print("No file selected")
    exit()

# ========== 2. Data reading ==========
data_raw = pd.read_excel(file_path)
X = data_raw.iloc[:, 0:5].values  # First 5 columns as features
y = data_raw.iloc[:, 5].values    # 6th column as target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}

# ========== 3. Define optimization problem ==========
class EnrOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        alpha_paras = x_decoded["alpha_paras"]
        l1_ratio_paras = x_decoded["l1_ratio_paras"]
        max_iter_paras = x_decoded["max_iter_paras"]

        enr = ElasticNet(alpha=alpha_paras, l1_ratio=l1_ratio_paras,
                         max_iter=max_iter_paras, random_state=1)
        enr.fit(self.data["X_train"], self.data["y_train"])
        y_predict = enr.predict(self.data["X_test"])
        mae = metrics.mean_absolute_error(self.data["y_test"], y_predict)
        return mae

# ========== 4. Set hyperparameter search space ==========
my_bounds = [
    FloatVar(lb=0.001, ub=1000., name="alpha_paras"),   # Regularization strength
    FloatVar(lb=0.0, ub=1.0, name="l1_ratio_paras"),    # L1/L2 ratio
    IntegerVar(lb=1, ub=5000, name="max_iter_paras")    # Maximum iterations
]

problem = EnrOptimizedProblem(bounds=my_bounds, minmax="min", data=data)

# ========== 5. Optimize with SMA ==========
model = SMA.OriginalSMA(epoch=100, pop_size=50)
model.solve(problem)

# Print best solution
print(f"Best agent: {model.g_best}")
print(f"Best solution: {model.g_best.solution}")
print(f"Best MAE: {model.g_best.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")
