import numpy as np
import pandas as pd
import warnings, os, random
warnings.filterwarnings("ignore")

# ============ Basic Dependencies ============
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
import joblib

from mealpy import FloatVar, Problem, SMA

# ============ 0. Set Seed & Configuration ============
SEED = 1
CV_FOLDS = 5                # K-fold Cross Validation on training set to avoid using the test set for hyperparameter search
SMA_EPOCHS = 15             # Number of iterations for SMA (adjustable based on time/computational power)
SMA_POP = 8                 # Population size for SMA (adjustable based on time/computational power)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)

# ============ 1. Select and Read Data (last column is target, all standardized) ============
def select_file():
    root = Tk()
    root.withdraw()
    return askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx;*.xls")])

file_path = select_file()
if not file_path:
    print("No file selected"); raise SystemExit

data_raw = pd.read_excel(file_path)
X = data_raw.iloc[:, :-1].values.astype(np.float32)
y = data_raw.iloc[:, -1].values.astype(np.float32).ravel()

# Strict 7:3 split (test set is only used for final evaluation, not for hyperparameter search)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

# ============ 2. Evaluation Function ============
def cv_mae_of_params(C, epsilon, gamma, Xtr, ytr, cv_folds=CV_FOLDS):
    """
    Performs K-fold cross-validation on the training set and returns the average MAE (the smaller, the better)
    """
    model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    # Sklearn's MAE score is negative (the higher, the better), so we negate it and calculate the mean
    scores = cross_val_score(
        model, Xtr, ytr,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        scoring="neg_mean_absolute_error",
        n_jobs=None
    )
    mae = -scores.mean()
    return mae

# ============ 3. Define SMA Optimization Problem ============
class SVRHyperProblem(Problem):
    """
    Search space uses 'log domain' for robust search (common practice):
      log10C    ∈ [-3, 3]     => C ∈ [1e-3, 1e3]
      log10gamma∈ [-4, 1]     => gamma ∈ [1e-4, 10]
      log10eps  ∈ [-4, -0.5]  => epsilon ∈ [1e-4, ~0.316]
    You can relax/narrow these ranges based on your needs.
    """
    def __init__(self, bounds=None, minmax="min", **kwargs):
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        log10C     = float(x_decoded["log10C"])
        log10gamma = float(x_decoded["log10gamma"])
        log10eps   = float(x_decoded["log10epsilon"])

        C       = 10.0 ** log10C
        gamma   = 10.0 ** log10gamma
        epsilon = 10.0 ** log10eps

        mae = cv_mae_of_params(C, epsilon, gamma, X_train, y_train, cv_folds=CV_FOLDS)
        return mae  # Goal: Minimize MAE

# Define continuous search space (log domain)
bounds = [
    FloatVar(lb=-3.0,  ub= 3.0,  name="log10C"),
    FloatVar(lb=-4.0,  ub= 1.0,  name="log10gamma"),
    FloatVar(lb=-4.0,  ub=-0.5,  name="log10epsilon"),
]

problem = SVRHyperProblem(bounds=bounds, minmax="min")

# ============ 4. Run SMA Hyperparameter Search ============
print("[Info] Start SMA hyperparameter search (SVR)...")
evolver = SMA.OriginalSMA(epoch=SMA_EPOCHS, pop_size=SMA_POP)
evolver.solve(problem)

best_vec = evolver.g_best.solution
best_decoded = evolver.problem.decode_solution(best_vec)
best_log10C, best_log10gamma, best_log10eps = (
    best_decoded["log10C"], best_decoded["log10gamma"], best_decoded["log10epsilon"]
)
best_C       = 10.0 ** best_log10C
best_gamma   = 10.0 ** best_log10gamma
best_epsilon = 10.0 ** best_log10eps

print(f"[Best CV-MAE] {evolver.g_best.target.fitness:.6f}")
print("[Best SVR params]")
print(f"  C       = {best_C:.6g}  (log10C={best_log10C:.3f})")
print(f"  gamma   = {best_gamma:.6g}  (log10gamma={best_log10gamma:.3f})")
print(f"  epsilon = {best_epsilon:.6g}  (log10epsilon={best_log10eps:.3f})")

# ============ 5. Retrain with Best Hyperparameters and Evaluate on Test Set ============
print("\n[Info] Retraining final SVR with best hyperparams on full training set...")
final_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma, epsilon=best_epsilon)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
test_mae = metrics.mean_absolute_error(y_test, y_pred)
print(f"[Result] Test MAE: {test_mae:.6f}")

# ============ 6. Save Model and Configuration ============
save_path = "svmr_best.joblib"
joblib.dump({
    "model": final_model,
    "best_params": {
        "C": best_C,
        "gamma": best_gamma,
        "epsilon": best_epsilon,
        "kernel": "rbf",
    },
    "seed": SEED,
    "cv_folds": CV_FOLDS,
    "sma": {"epochs": SMA_EPOCHS, "pop_size": SMA_POP},
    "n_features": X_train.shape[1],
}, save_path)
print(f"[Saved] Final model saved to: {os.path.abspath(save_path)}")
