"""
This script allows the user to select an Excel file via a Tkinter file dialog, 
reads the data into a pandas DataFrame, computes basic descriptive statistics 
(mean, standard deviation, min, quartiles, median, max), and prints the results.
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window and hide it
root = tk.Tk()
root.withdraw()  # Hide the main window

# Open a file selection dialog
file_path = filedialog.askopenfilename(
    title="Select an Excel file",
    filetypes=[("Excel Files", "*.xlsx;*.xls")]
)

# If the user selects a file
if file_path:
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Compute descriptive statistics
    statistics = {
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'Q1': df.quantile(0.25),
        'median': df.median(),
        'Q3': df.quantile(0.75),
        'max': df.max()
    }

    # Convert to DataFrame for better visualization
    statistics_df = pd.DataFrame(statistics)

    # Print the results
    print(statistics_df)
else:
    print("No file selected")

