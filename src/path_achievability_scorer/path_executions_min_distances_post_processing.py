import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

def plot_min_distances(scene_id, saved_scores_dir, d_obstacle_offset):

    # Load the data
    csv_file = os.path.join(saved_scores_dir, f"scene_{scene_id}", f"scene_{scene_id}_experiment_execution_results.csv")
    # Specify 'Nan' and 'NaN' as missing values
    df = pd.read_csv(csv_file, na_values=['Nan', 'NaN'])

    # Remove experiments where all columns except experiment_id are NaN
    df = df.dropna(subset=[col for col in df.columns if col != 'experiment_id'], how='all')

    # Convert min_distance from meters to mm
    df['min_distance'] = df['min_distance'] * 1000  # Convert to mm

    # Remove rows where min_distance or success is NaN
    df_clean = df.dropna(subset=['min_distance', 'success'])

    # Ensure 'success' is numeric
    df_clean['success'] = pd.to_numeric(df_clean['success'], errors='coerce').astype(int)

    # Prepare data for plotting
    exp_numbers = df_clean['experiment_id'].values
    min_distances = df_clean['min_distance'].values
    successes = df_clean['success'].values.astype(int)

    # Create the figure
    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust size as needed

    fig_title = f"Scene {scene_id} Executions Minimum Distances"
    ax.set_title(fig_title, fontsize=35)
    ax.set_xlabel("Experiment Number", fontsize=30)
    ax.set_ylabel("Minimum Distance to Obstacles (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Adjust x-axis ticks
    ax.set_xticks(exp_numbers)
    ax.set_xticklabels(['' if i%5 != 0 else str(int(i)) for i in exp_numbers], rotation=90)

    # Add vertical grid lines at each experiment number
    ax.grid(True, axis='x', linestyle='-', linewidth=0.5)

    # Plot the min distance
    # If the success is 1, plot the min distance in blue
    # If the success is 0, plot the min distance in red
    success_mask = successes == 1
    failure_mask = successes == 0

    ax.plot(exp_numbers[success_mask], min_distances[success_mask], 'bo', label='Min Distance (Success)')
    ax.plot(exp_numbers[failure_mask], min_distances[failure_mask], 'ro', label='Min Distance (Fail)')

    # Add a horizontal line at y=d_obstacle_offset
    ax.axhline(y=d_obstacle_offset, color='g', linestyle='--', linewidth=2)
    # Shade the area below y=d_obstacle_offset until y=0
    ax.axhspan(ymin=0, ymax=d_obstacle_offset, facecolor='orange', alpha=0.3)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    # Shade the area below y=0
    ax.axhspan(ymin=ax.get_ylim()[0], ymax=0, facecolor='red', alpha=0.3)

    # Bring grid lines to the front
    ax.set_axisbelow(False)

    # Add legend explaining the annotations
    # Create custom legend handles
    min_distance_point_success = Line2D([], [], color='blue', marker='o', linestyle='None', label='Min Distance (Success)')
    min_distance_point_fail = Line2D([], [], color='red', marker='o', linestyle='None', label='Min Distance (Fail)')

    # Add the legend to your plot
    ax.legend(handles=[min_distance_point_success, min_distance_point_fail], fontsize=20)

    # Save the figure
    plots_dir = '.'  # Current directory
    plot_file = os.path.join(plots_dir, f'scene_{scene_id}_execution_min_distances.png')

    # Also add "_10_segments" to the file name if saved_scores_dir contains "10_segments"
    if '10_segments' in saved_scores_dir:
        plot_file = plot_file.replace('.png', '_10_segments.png')

    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    plt.close(fig)

# ------------------------------------------------------------------------
d_obstacle_offset = 2  # mm
scenes = [1, 2, 3, 4]

# Set the directory containing the CSV files
saved_scores_dir = '../tesseract_planner/generated_plans_i9_10885h'
for scene_id in scenes:
    plot_min_distances(scene_id, saved_scores_dir, d_obstacle_offset)

# For the 10_segments data, uncomment the line below
saved_scores_dir = '../tesseract_planner/generated_plans_i9_10885h_10_segments'
for scene_id in scenes:
    plot_min_distances(scene_id, saved_scores_dir, d_obstacle_offset)