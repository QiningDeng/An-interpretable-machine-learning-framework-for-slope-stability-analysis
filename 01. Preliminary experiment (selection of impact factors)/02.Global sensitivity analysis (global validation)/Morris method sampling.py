"""
This script performs Morris sampling using the SALib library.
It defines the parameter space, generates Morris samples, 
and saves the sampled results to an Excel file.
"""

import numpy as np
import pandas as pd
from SALib.sample import morris

# Define parameter information
factor_names = ['x1','x2','x3','y1','y2','z','gamma','c0','phi','E','v','psi']

ranges_morris = pd.DataFrame({
    'min': [1, 1, 1, 1, 1, 1, 10, 0, 0, 1e3, 0.1, 0],
    'max': [20, 20, 20, 20, 20, 20, 30, 50, 50, 1e8, 0.5, 0.3]
}, index=factor_names)

# Construct the SALib problem definition
problem = {
    'num_vars': len(factor_names),
    'names': factor_names,
    'bounds': [[ranges_morris.loc[name, 'min'], 
               ranges_morris.loc[name, 'max']] for name in factor_names]
}

# Perform Morris sampling
param_values = morris.sample(
    problem=problem,
    N=100,           # Number of trajectories
    num_levels=12,   # Grid resolution level
)

# Convert samples to DataFrame and save to Excel
samples_df = pd.DataFrame(param_values, columns=factor_names)
samples_df.to_excel('morris_samples.xlsx', index=False)

print("Sampling completed. Samples saved to morris_samples.xlsx")
