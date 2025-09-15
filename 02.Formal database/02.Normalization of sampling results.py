"""
This script reads an existing Excel file containing sampled data, 
applies Min-Max normalization to all columns using sklearn's MinMaxScaler, 
and saves the normalized results into a new Excel file.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the previously generated Excel file
input_filename = 'lhs_sampled_data.xlsx'
data = pd.read_excel(input_filename)

# Create a MinMaxScaler object for normalization
scaler = MinMaxScaler()

# Apply Min-Max normalization to all columns
normalized_data = scaler.fit_transform(data)

# Convert the normalized data back into a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

# Save the normalized data into a new Excel file
output_filename = 'normalized_sampled_data.xlsx'
normalized_df.to_excel(output_filename, index=False)

print(f"Normalized data has been successfully saved to {output_filename}")
