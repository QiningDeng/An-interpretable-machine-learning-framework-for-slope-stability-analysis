"""
This script performs Morris sensitivity analysis.
It calculates the absolute mean of elementary effects (μ*) and the standard deviation (σ),
categorizes variables into quadrants based on reference thresholds, 
and generates a Morris plot with quadrant annotations and reference lines.
Results are saved both as a plot (PNG) and a CSV file containing statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import filedialog

def analyze_morris_sensitivity(file_path, output_path):
    """
    Morris sensitivity analysis: calculate the absolute mean of elementary effects (μ*) 
    and standard deviation (σ), and generate a Morris plot with reference dashed lines.
    
    Args:
        file_path (str): Path to the Excel input file
        output_path (str): Directory path to save results and plots
    
    Returns:
        tuple: (DataFrame with statistics, matplotlib figure object)
    """
    # 1. Read Excel data and extract variable names
    df = pd.read_excel(file_path, header=0)
    print(f"Data successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Extract variable names (skip index column and target column)
    input_columns = df.columns[1:-1].tolist()
    
    # 2. Parameter settings
    n_trajectories = 100    # Number of trajectories
    n_steps_per_traj = 13   # Steps per trajectory (12 variables + starting point)
    
    # 3. Reshape data into trajectory structure (100, 13, number_of_columns)
    data = df.values
    trajectories = data.reshape((n_trajectories, n_steps_per_traj, data.shape[1]))
    
    # 4. Prepare storage for Elementary Effects (EE) of each variable
    ee_data = {var: [] for var in input_columns}
    
    # 5. Iterate through each trajectory and compute elementary effects
    for traj in trajectories:
        for step in range(n_steps_per_traj - 1):
            # Get current and next point
            point1 = traj[step, 1:-1]    # Input variables
            point2 = traj[step + 1, 1:-1]
            y1 = traj[step, -1]          # Target output
            y2 = traj[step + 1, -1]
            
            # Compute the change in input variables
            delta = point2 - point1
            
            # Find the index of the changed variable
            non_zero_idx = np.where(np.abs(delta) > 1e-5)[0]
            
            # Ensure only one variable changes
            if len(non_zero_idx) != 1:
                continue
                
            var_idx = non_zero_idx[0]
            var_name = input_columns[var_idx]
            delta_x = delta[var_idx]
            delta_y = y2 - y1
            
            # Compute elementary effect (EE)
            ee = delta_y / np.abs(delta_x)
            
            # Store EE for the corresponding variable
            ee_data[var_name].append(ee)
    
    # 6. Compute statistics for each variable
    stats = []
    all_mu_star = []
    all_sigma = []
    
    for var, values in ee_data.items():
        if values:
            abs_mean = np.mean(np.abs(values))  # μ*: Absolute mean of EE
            std_ee = np.std(values, ddof=1)     # σ: Standard deviation of EE
            stats.append({
                'Variable': var,
                'μ* (Abs Mean EE)': abs_mean,
                'σ (Std EE)': std_ee
            })
            all_mu_star.append(abs_mean)
            all_sigma.append(std_ee)
    
    stats_df = pd.DataFrame(stats)
    
    # 7. Compute reference line positions (use mean values as thresholds)
    mu_star_mean = np.mean(all_mu_star)
    sigma_mean = np.mean(all_sigma)
    
    # 8. Create Morris plot with reference dashed lines
    plt.figure(figsize=(10, 8))
    
    # Set global font to Times New Roman
    plt.rcParams.update({'font.family': 'Times New Roman'})
    
    # Define RGB colors for four quadrants
    colors_dict = {
        'I': (54/255, 80/255, 131/255),   # Quadrant I
        'II': (183/255, 131/255, 175/255),# Quadrant II
        'III': (245/255, 166/255, 115/255),# Quadrant III
        'IV': (252/255, 219/255, 114/255) # Quadrant IV
    }
    
    # Assign quadrant colors for each variable
    quadrants = []
    for i, row in stats_df.iterrows():
        mu_star = row['μ* (Abs Mean EE)']
        sigma = row['σ (Std EE)']
        
        if mu_star >= mu_star_mean and sigma >= sigma_mean:
            quadrant = 'I'
        elif mu_star >= mu_star_mean and sigma < sigma_mean:
            quadrant = 'IV'
        elif mu_star < mu_star_mean and sigma >= sigma_mean:
            quadrant = 'II'
        else:
            quadrant = 'III'
        
        quadrants.append(quadrant)
    
    # Add quadrant classification to DataFrame
    stats_df['Quadrant'] = quadrants
    
    # Scatter plot with colors based on quadrants
    ax = sns.scatterplot(
        data=stats_df, 
        x='μ* (Abs Mean EE)', 
        y='σ (Std EE)', 
        s=120, hue='Quadrant', 
        palette=colors_dict,
        edgecolor='black', linewidth=0.8
    )
    
    # Add reference dashed lines
    plt.axvline(x=mu_star_mean, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=sigma_mean, color='gray', linestyle='--', alpha=0.7)
    
    # Add threshold labels
    plt.text(mu_star_mean * 1.02, plt.ylim()[1] * 0.95, 
             f'The average of μ* is {mu_star_mean:.4f}', 
             color='gray', fontsize=16, rotation=0)
    plt.text(plt.xlim()[1] * 0.95, sigma_mean * 1.02, 
             f'The average of σ is {sigma_mean:.4f}', 
             color='gray', fontsize=16, ha='right', rotation=0)
    
    # Add quadrant labels
    plt.text(plt.xlim()[1] * 0.98, plt.ylim()[1] * 0.98, 'I', 
             fontsize=16, color=colors_dict['I'], ha='right', va='top', weight='bold')
    plt.text(plt.xlim()[0] * 1.02, plt.ylim()[1] * 0.98, 'II', 
             fontsize=16, color=colors_dict['II'], ha='left', va='top', weight='bold')
    plt.text(plt.xlim()[0] * 1.02, plt.ylim()[0] * 1.02, 'III', 
             fontsize=16, color=colors_dict['III'], ha='left', va='bottom', weight='bold')
    plt.text(plt.xlim()[1] * 0.98, plt.ylim()[0] * 1.02, 'IV', 
             fontsize=16, color=colors_dict['IV'], ha='right', va='bottom', weight='bold')
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Quadrant I: High μ*, High σ',
               markerfacecolor=colors_dict['I'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Quadrant II: Low μ*, High σ',
               markerfacecolor=colors_dict['II'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Quadrant III: Low μ*, Low σ',
               markerfacecolor=colors_dict['III'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Quadrant IV: High μ*, Low σ',
               markerfacecolor=colors_dict['IV'], markersize=10),
    ]
    
    plt.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.8, fontsize=16)
    
    # Beautify the plot
    plt.xlabel('Absolute Mean of Elementary Effects (μ*)', fontsize=16)
    plt.ylabel('Standard Deviation of Elementary Effects (σ)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Adjust axis line width
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.2 * spine.get_linewidth())
    
    # Set tick label size
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Save plot
    morris_plot = plt.gcf()  
    morris_plot.savefig(output_path + '/morris_sensitivity_plot_with_quadrants.png', dpi=1000, bbox_inches='tight')
    print(f"Plot saved as '{output_path}/morris_sensitivity_plot_with_quadrants.png'")
    
    # Save results
    stats_df.to_csv(output_path + '/morris_sensitivity_stats_with_quadrants.csv', index=False)
    print(f"Statistics saved as '{output_path}/morris_sensitivity_stats_with_quadrants.csv'")
    
    return stats_df, morris_plot

# Usage example
if __name__ == "__main__":
    # Open file dialog to select Excel file
    root = tk.Tk()
    root.withdraw()  # Hide main window
    excel_file = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx;*.xls")])
    
    # Open file dialog to select output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    
    # Run analysis
    stats_df, morris_plot = analyze_morris_sensitivity(excel_file, output_folder)
    
    # Print statistics
    print("\nMorris sensitivity analysis results (with quadrant classification):")
    print(stats_df.to_string(index=False))
    
    # Show plot
    plt.show()
