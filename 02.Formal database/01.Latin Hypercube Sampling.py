"""
This script generates a large number of samples for five geotechnical feature factors 
using Latin Hypercube Sampling (LHS). The sampled values are scaled to their 
corresponding ranges and saved into an Excel file for further use.
"""

import pandas as pd
from pyDOE import lhs

# Define the ranges of five feature factors
H2_range = (1, 20)   # Range of H2
c_range = (0, 30)    # Range of cohesion c
ϕ_range = (0, 50)    # Range of friction angle ϕ
W_range = (1, 20)    # Range of width W
γ_range = (10, 30)   # Range of unit weight γ

# Define the number of samples
sample_size = 100000

# Generate samples using Latin Hypercube Sampling
lhs_samples = lhs(5, samples=sample_size)

# Scale each feature factor to its corresponding range
H2 = H2_range[0] + (H2_range[1] - H2_range[0]) * lhs_samples[:, 0]
c = c_range[0] + (c_range[1] - c_range[0]) * lhs_samples[:, 1]
ϕ = ϕ_range[0] + (ϕ_range[1] - ϕ_range[0]) * lhs_samples[:, 2]
W = W_range[0] + (W_range[1] - W_range[0]) * lhs_samples[:, 3]
γ = γ_range[0] + (γ_range[1] - γ_range[0]) * lhs_samples[:, 4]

# Organize the results into a DataFrame
data = pd.DataFrame({
    'H2': H2,
    'c': c,
    'ϕ': ϕ,
    'W': W,
    'γ': γ
})

# Save the data into an Excel file
output_filename = 'sampled_data.xlsx'
data.to_excel(output_filename, index=False)

print(f"Data has been successfully saved to {output_filename}")
