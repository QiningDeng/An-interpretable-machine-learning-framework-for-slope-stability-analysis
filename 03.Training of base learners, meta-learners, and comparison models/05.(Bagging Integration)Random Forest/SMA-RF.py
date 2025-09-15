import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Import the Random Forest Regressor model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mealpy import IntegerVar, SMA, Problem
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Popup file selection dialog, allowing the user to select an Excel file
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx;*.xls")])
    return file_path

# Read the Excel file
file_path = select_file()  # Select the file
if file_path:
    # Read the Excel file
    data = pd.read_excel(file_path)
    
    # Data preprocessing, assuming the first five columns are features, and the sixth column is the target
    X = data.iloc[0:, 0:5].values  # Extract feature data
    y = data.iloc[0:, 5].values    # Extract target data
    
    # Split the data into training and test sets with a 70:30 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    # Define the optimization problem
    class RfOptimizedProblem(Problem):
        def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
            self.data = data
            super().__init__(bounds, minmax, **kwargs)

        def obj_func(self, x):
            x_decoded = self.decode_solution(x)
            n_estimators_paras, max_depth_paras, min_samples_split_paras = x_decoded["n_estimators_paras"], x_decoded["max_depth_paras"], x_decoded["min_samples_split_paras"]

            # Initialize the Random Forest Regressor model
            rf = RandomForestRegressor(n_estimators=n_estimators_paras, max_depth=max_depth_paras, min_samples_split=min_samples_split_paras, random_state=1)
            
            # Fit the model
            rf.fit(self.data["X_train"], self.data["y_train"])
            
            # Predict
            y_predict = rf.predict(self.data["X_test"])
            
            # Calculate Mean Absolute Error (MAE)
            mae = metrics.mean_absolute_error(self.data["y_test"], y_predict)
            return mae  # The objective function is MAE, and we aim to minimize MAE

    # Set the search range for hyperparameters, including Random Forest parameters
    my_bounds = [
        IntegerVar(lb=10, ub=100, name="n_estimators_paras"),  # n_estimators
        IntegerVar(lb=1, ub=10, name="max_depth_paras"),        # max_depth
        IntegerVar(lb=2, ub=50, name="min_samples_split_paras")  # min_samples_split
    ]

    # Initialize the problem instance
    problem = RfOptimizedProblem(bounds=my_bounds, minmax="min", data=data)

    # Initialize the SMA algorithm and perform optimization
    model = SMA.OriginalSMA(epoch=5, pop_size=5)
    model.solve(problem)

    # Output the optimization results
    print(f"Best agent: {model.g_best}")
    print(f"Best solution: {model.g_best.solution}")
    print(f"Best MAE: {model.g_best.target.fitness}")
    print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")

else:
    print("No file selected")
