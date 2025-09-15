"""
This script allows the user to select an Excel file via a Tkinter file dialog, 
extracts the first five columns as input features, computes the Pearson 
correlation matrix, and visualizes it as a heatmap with a custom color map. 
The heatmap is styled with Times New Roman fonts, includes labeled colorbar, 
and is saved as a high-resolution PNG file.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib.colors import LinearSegmentedColormap

# Create a Tkinter window and hide it
root = Tk()
root.withdraw()  # Hide the root window

# Open a file selection dialog
file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx;*.xls")])

# Check if the user selected a file
if file_path:
    # Import the Excel file
    df = pd.read_excel(file_path)

    # Get the first five columns as input features
    input_features = df.iloc[:, :5]

    # Compute the Pearson correlation matrix
    correlation_matrix = input_features.corr(method='pearson')

    # Define custom colors
    colors = [(54/255, 80/255, 131/255),    # 0: RGB(54, 80, 131)
              (155/255, 107/255, 157/255),  # 0.25: RGB(155, 107, 157)
              (183/255, 131/255, 175/255),  # 0.5: RGB(183, 131, 175)
              (245/255, 166/255, 115/255),  # 0.75: RGB(245, 166, 115)
              (252/255, 219/255, 114/255)]  # 1: RGB(252, 219, 114)

    # Create a LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Create a figure and plot the correlation heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        correlation_matrix, annot=True, cmap=custom_cmap, fmt='.2f', cbar=True,
        xticklabels=input_features.columns, yticklabels=input_features.columns,
        annot_kws={'size': 12, 'weight': 'normal', 'family': 'Times New Roman'}, linewidths=0.5
    )

    # Set font for x and y ticks
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')

    # Get the colorbar object
    cbar = ax.collections[0].colorbar

    # Customize colorbar ticks and labels
    cbar.ax.tick_params(labelsize=12, labelcolor='black', direction='in', labelrotation=0)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(12)

    # Add colorbar label
    cbar.set_label('Pearson Correlation Coefficient', fontsize=12, fontname='Times New Roman')

    # Save the figure with high DPI
    plt.tight_layout()
    plt.savefig("correlation_heatmap_custom.png", dpi=1000)  # Save the figure

    # Show the heatmap
    plt.show()
else:
    print("No file selected.")
