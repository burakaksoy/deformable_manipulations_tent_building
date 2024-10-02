import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import re
import itertools

def plot_scene_experiments(scene_id, dlo_type, saved_scores_dir):
    """
    Plots all experiments data and statistics for a given scene.

    Args:
        scene_id (int): Scene ID.
        saved_scores_dir (str): Directory containing the saved path scores.
    """
    # Initialize data structures
    experiments_data = {}  # Key: experiment_number, value: data
    smoothed_avr_errs = []
    avr_err_changes_on_smoothed = []
    min_distance_values = []
    
    if dlo_type is None:
        scene_dir = os.path.join(saved_scores_dir, f'scene_{scene_id}')
    else:
        scene_dir = os.path.join(saved_scores_dir, f'scene_{scene_id}_dlo_{dlo_type}')
        
    # Directory containing the scores
    scores_dir = os.path.join(scene_dir, 'scores')
    
    # Get list of pickle files
    pickle_files = [f for f in os.listdir(scores_dir) if f.endswith('_achievability_scores.pkl')]
    
    for pkl_file in pickle_files:
        # Extract experiment_number from filename
        if dlo_type is None:
            match = re.match(f'scene_{scene_id}_experiment_(\\d+)_achievability_scores.pkl', pkl_file)
        else:
            match = re.match(f'scene_{scene_id}_dlo_{dlo_type}_experiment_(\\d+)_achievability_scores.pkl', pkl_file)
            
        if match:
            experiment_number = int(match.group(1))
            pkl_path = os.path.join(scores_dir, pkl_file)
            # Load the pickle file
            with open(pkl_path, 'rb') as f:
                scores_all = pickle.load(f)
            scores, scores_min_distances = scores_all
            # Use the smoothed data
            if len(scores) == 17:
                # Unpack the scores
                (errs,
                peak_err,
                peak_err_waypoint_idx,
                avr_err,

                err_changes,
                peak_err_change,
                peak_err_change_idx,
                avr_err_change,

                smoothed_errs,
                smoothed_peak_err,
                smoothed_peak_err_waypoint_idx,
                smoothed_avr_err,

                err_changes_on_smoothed,
                peak_err_change_on_smoothed,
                peak_err_change_idx_on_smoothed,
                avr_err_change_on_smoothed,

                scoring_duration_per_waypoint) = scores

                # Unpack scores_min_distances
                (min_distances,
                min_distance_value,
                min_distance_idx) = scores_min_distances

                # Store data
                experiments_data[experiment_number] = {
                    'smoothed_errs': smoothed_errs,
                    'smoothed_peak_err': smoothed_peak_err,
                    'smoothed_peak_err_waypoint_idx': smoothed_peak_err_waypoint_idx,
                    'smoothed_avr_err': smoothed_avr_err,

                    'err_changes_on_smoothed': err_changes_on_smoothed,
                    'peak_err_change_on_smoothed': peak_err_change_on_smoothed,
                    'peak_err_change_idx_on_smoothed': peak_err_change_idx_on_smoothed,
                    'avr_err_change_on_smoothed': avr_err_change_on_smoothed,

                    'min_distances': min_distances,
                    'min_distance_value': min_distance_value,
                    'min_distance_idx': min_distance_idx,

                    'errs': errs,
                    'err_changes': err_changes,
                    'experiment_number': experiment_number,
                }

                # Collect average errors and other metrics
                smoothed_avr_errs.append(smoothed_avr_err)
                avr_err_changes_on_smoothed.append(avr_err_change_on_smoothed)
                min_distance_values.append(min_distance_value)
            else:
                print(f"Scores in {pkl_file} does not have expected length 17.")
        else:
            print(f"Filename {pkl_file} does not match expected pattern.")
    
    # Now, proceed to plot
    # First, compute min and max average errors, error changes, min distances for alpha mapping
    smoothed_avr_errs_array = np.array(smoothed_avr_errs)
    avr_err_changes_on_smoothed_array = np.array(avr_err_changes_on_smoothed)
    min_distance_values_array = np.array(min_distance_values)
    
    min_avr_err = smoothed_avr_errs_array.min()
    max_avr_err = smoothed_avr_errs_array.max()
    min_avr_err_change = avr_err_changes_on_smoothed_array.min()
    max_avr_err_change = avr_err_changes_on_smoothed_array.max()
    min_min_distance = min_distance_values_array.min()
    max_min_distance = min_distance_values_array.max()
    
    # Set alpha values
    alpha_min = 0.2
    alpha_max = 1.0

    # Create the figure
    # fig, axs = plt.subplots(3, 2, figsize=(32, 32))
    fig, axs = plt.subplots(3, 2, figsize=(32, 32), gridspec_kw={'width_ratios': [1, 2]})
    
    if dlo_type is None:
        fig_title = f"Scene {scene_id} Experiments Path Analysis"
    else:
        fig_title = f"Scene {scene_id} DLO {dlo_type} Experiments Path Analysis"
        
    # Adjust top margin to prevent title overlap
    plt.subplots_adjust(top=0.92)
    fig.suptitle(fig_title, fontsize=40, y=0.98)

    # Column 1, Row 1: Error vs Waypoints
    ax = axs[0, 0]
    for exp_num, data in experiments_data.items():
        smoothed_errs = data['smoothed_errs']
        smoothed_avr_err = data['smoothed_avr_err']
        smoothed_peak_err = data['smoothed_peak_err']
        smoothed_peak_err_waypoint_idx = data['smoothed_peak_err_waypoint_idx']
        # Map average error to alpha
        if max_avr_err - min_avr_err > 0:
            alpha = alpha_min + (smoothed_avr_err - min_avr_err) / (max_avr_err - min_avr_err) * (alpha_max - alpha_min)
        else:
            alpha = alpha_max
        ax.plot(smoothed_errs, alpha=alpha)
        # Mark the peak error point
        ax.plot(smoothed_peak_err_waypoint_idx, smoothed_peak_err, 'o', alpha=alpha)
        
        # # Annotate with experiment ID
        # ax.annotate(f'{exp_num}', (smoothed_peak_err_waypoint_idx, smoothed_peak_err))
        
        # Annotate with waypoint index, positioning the text below the data point
        ax.annotate(f'{exp_num}',
                    xy=(smoothed_peak_err_waypoint_idx, smoothed_peak_err),
                    xytext=(0, 10),  # Offset text 10 points above the data point
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='top')
        
    ax.set_title("Error vs Waypoint Index", fontsize=35)
    ax.set_ylabel("Error (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # # Add legend explaining the annotations
    # ax.legend(['Smoothed Errors', 'Peak Error Points'], fontsize=20)
    
    # ax.text(0.95, 0.95, 'Numbers: Experiment IDs', transform=ax.transAxes, fontsize=15,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Column 1, Row 2: Error Change vs Waypoints
    ax = axs[1, 0]
    for exp_num, data in experiments_data.items():
        err_changes_on_smoothed = data['err_changes_on_smoothed']
        avr_err_change_on_smoothed = data['avr_err_change_on_smoothed']
        peak_err_change_on_smoothed = data['peak_err_change_on_smoothed']
        peak_err_change_idx_on_smoothed = data['peak_err_change_idx_on_smoothed']
        # Map average error change to alpha
        if max_avr_err_change - min_avr_err_change > 0:
            alpha = alpha_min + (avr_err_change_on_smoothed - min_avr_err_change) / (max_avr_err_change - min_avr_err_change) * (alpha_max - alpha_min)
        else:
            alpha = alpha_max
        ax.plot(err_changes_on_smoothed, alpha=alpha)
        # Mark the peak error change point
        ax.plot(peak_err_change_idx_on_smoothed, peak_err_change_on_smoothed, 'o', alpha=alpha)
        
        # # Annotate with experiment ID
        # ax.annotate(f'{exp_num}', (peak_err_change_idx_on_smoothed, peak_err_change_on_smoothed))
        
        # Annotate with waypoint index, positioning the text below the data point
        ax.annotate(f'{exp_num}',
                    xy=(peak_err_change_idx_on_smoothed, peak_err_change_on_smoothed),
                    xytext=(0, 10),  # Offset text 10 points above the data point
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='top')
        
    ax.set_title("Error Change vs Waypoint Index", fontsize=35)
    ax.set_ylabel("Error Change (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # # Add legend explaining the annotations
    # ax.legend(['Error Changes', 'Peak Error Change Points'], fontsize=20)
    
    # ax.text(0.95, 0.95, 'Numbers: Experiment IDs', transform=ax.transAxes, fontsize=15,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Column 1, Row 3: Minimum Distance vs Waypoints
    ax = axs[2, 0]
    for exp_num, data in experiments_data.items():
        min_distances = data['min_distances']
        min_distance_value = data['min_distance_value']
        min_distance_idx = data['min_distance_idx']
        # Map min distance to alpha (reverse mapping)
        if max_min_distance - min_min_distance > 0:
            alpha = alpha_min + (max_min_distance - min_distance_value) / (max_min_distance - min_min_distance) * (alpha_max - alpha_min)
        else:
            alpha = alpha_max
        ax.plot(min_distances, alpha=alpha)
        # Mark the min distance point
        ax.plot(min_distance_idx, min_distance_value, 'o', alpha=alpha)
        
        # # Annotate with experiment ID
        # ax.annotate(f'{exp_num}', (min_distance_idx, min_distance_value))
        
        # Annotate with waypoint index, positioning the text below the data point
        ax.annotate(f'{exp_num}',
                    xy=(min_distance_idx, min_distance_value),
                    xytext=(0, -5),  # Offset text 5 points below the data point
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='top')
        
        
    ax.set_title("Minimum Distance vs Waypoint Index", fontsize=35)
    ax.set_xlabel("Waypoint Index", fontsize=30)
    ax.set_ylabel("Minimum Distance to Obstacles (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    # Shade the area below y=0
    ax.axhspan(ymin=ax.get_ylim()[0], ymax=0, facecolor='red', alpha=0.3)
    
    # # Add legend explaining the annotations
    # ax.legend(['Minimum Distances', 'Minimum Distance Points'], fontsize=20)
    
    # ax.text(0.95, 0.95, 'Numbers: Experiment IDs', transform=ax.transAxes, fontsize=15,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Column 2, Row 1: Error Statistics vs Experiment Numbers (Boxplots)
    ax = axs[0,1]
    # Collect errors per experiment
    all_smoothed_errs = [data['smoothed_errs'] for data in experiments_data.values()]
    exp_numbers = list(experiments_data.keys())
    exp_numbers.sort()
    # Create boxplot
    boxplot = ax.boxplot(all_smoothed_errs, positions=exp_numbers, widths=0.6,
                         patch_artist=True, 
                         showfliers=False, flierprops={'markersize': 2, 'markerfacecolor': 'gray', 'alpha': 0.3},
                         medianprops={'color': 'blue', 'linewidth': 2},
                         boxprops={'facecolor':'lightgray', 'alpha':0.5},
                         whiskerprops={'linewidth':0.5, 'linestyle':'--'},
                         showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2, 'linestyle':'-'},
                         showcaps=True, capprops={'linewidth':0.5})
    ax.set_title("Error Statistics vs Experiment Numbers", fontsize=35)
    ax.set_ylabel("Error (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # Adjust x-axis ticks
    ax.set_xticks(exp_numbers)
    ax.set_xticklabels(['' if i%5 != 0 else str(i) for i in exp_numbers], rotation=90)
    
    # Mark the peak errors
    for data in experiments_data.values():
        exp_num = data['experiment_number']
        smoothed_peak_err = data['smoothed_peak_err']
        smoothed_peak_err_waypoint_idx = data['smoothed_peak_err_waypoint_idx']
        # Plot the peak error
        ax.plot(exp_num, smoothed_peak_err, 'ro')
        
        # # Annotate with waypoint index (without 'WP')
        # ax.annotate(f'{smoothed_peak_err_waypoint_idx}', (exp_num, smoothed_peak_err))
        
        # Annotate with waypoint index, positioning the text below the data point
        ax.annotate(f'{smoothed_peak_err_waypoint_idx}',
                    xy=(exp_num, smoothed_peak_err),
                    xytext=(0, 10),  # Offset text 10 points above the data point
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='top')
        
    # # Add legend explaining the annotations
    # Create custom legend handles
    median_line = Line2D([], [], color='blue', linewidth=2, label='Median')
    mean_line = Line2D([], [], color='red', linewidth=2, linestyle='-', label='Mean')
    max_error_point = Line2D([], [], color='red', marker='o', linestyle='None', label='Max Error')

    # Add the legend to your plot
    ax.legend(handles=[median_line, mean_line, max_error_point], loc='lower left', fontsize=30)
    
    # ax.text(0.95, 0.95, 'Numbers: Waypoint Indices', transform=ax.transAxes, fontsize=15,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Column 2, Row 2: Error Change Statistics vs Experiment Numbers (Boxplots)
    ax = axs[1,1]
    # Collect error changes per experiment
    all_err_changes_on_smoothed = [data['err_changes_on_smoothed'] for data in experiments_data.values()]
    # Create boxplot
    boxplot = ax.boxplot(all_err_changes_on_smoothed, positions=exp_numbers, widths=0.6,
                         patch_artist=True, 
                         showfliers=False, flierprops={'markersize': 2, 'markerfacecolor': 'gray', 'alpha': 0.3},
                         medianprops={'color': 'blue', 'linewidth': 2},
                         boxprops={'facecolor':'lightgray', 'alpha':0.5},
                         whiskerprops={'linewidth':0.5, 'linestyle':'--'},
                         showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2, 'linestyle':'-'},
                         showcaps=True, capprops={'linewidth':0.5})
    ax.set_title("Error Change Statistics vs Experiment Numbers", fontsize=35)
    ax.set_ylabel("Error Change (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # Adjust x-axis ticks
    ax.set_xticks(exp_numbers)
    ax.set_xticklabels(['' if i%5 != 0 else str(i) for i in exp_numbers], rotation=90)
    # Mark the peak error changes
    for data in experiments_data.values():
        exp_num = data['experiment_number']
        peak_err_change_on_smoothed = data['peak_err_change_on_smoothed']
        peak_err_change_idx_on_smoothed = data['peak_err_change_idx_on_smoothed']
        # Plot the peak error change
        ax.plot(exp_num, peak_err_change_on_smoothed, 'ro')
        
        # Annotate with waypoint index (without 'WP')
        # ax.annotate(f'{peak_err_change_idx_on_smoothed}', (exp_num, peak_err_change_on_smoothed))
        
        # Annotate with waypoint index, positioning the text below the data point
        ax.annotate(f'{peak_err_change_idx_on_smoothed}',
                    xy=(exp_num, peak_err_change_on_smoothed),
                    xytext=(0, 10),  # Offset text 10 points above the data point
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='top')
        
    # # Add legend explaining the annotations
    # Create custom legend handles
    median_line = Line2D([], [], color='blue', linewidth=2, label='Median')
    mean_line = Line2D([], [], color='red', linewidth=2, linestyle='-', label='Mean')
    max_err_increase_point = Line2D([], [], color='red', marker='o', linestyle='None', label='Max Error Increase')

    # Add the legend to your plot
    ax.legend(handles=[median_line, mean_line, max_err_increase_point], loc='lower left', fontsize=30)
    
    # ax.text(0.95, 0.95, 'Numbers: Waypoint Indices', transform=ax.transAxes, fontsize=15,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Column 2, Row 3: Minimum Distance Statistics vs Experiment Numbers (Boxplots)
    ax = axs[2,1]
    # Collect min distances per experiment
    all_min_distances = [data['min_distances'] for data in experiments_data.values()]
    # Create boxplot
    boxplot = ax.boxplot(all_min_distances, positions=exp_numbers, widths=0.6,
                         patch_artist=True, 
                         showfliers=False, flierprops={'markersize': 2, 'markerfacecolor': 'gray', 'alpha': 0.3},
                         medianprops={'color': 'blue', 'linewidth': 2},
                         boxprops={'facecolor':'lightgray', 'alpha':0.5},
                         whiskerprops={'linewidth':0.5, 'linestyle':'--'},
                         showmeans=True, meanline=True, meanprops={'color':'red', 'linewidth':2, 'linestyle':'-'},
                         showcaps=True, capprops={'linewidth':0.5})
    ax.set_title("Minimum Distance Statistics vs Experiment Numbers", fontsize=35)
    ax.set_xlabel("Experiment Number", fontsize=30)
    ax.set_ylabel("Minimum Distance to Obstacles (mm)", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # Adjust x-axis ticks
    ax.set_xticks(exp_numbers)
    ax.set_xticklabels(['' if i%5 != 0 else str(i) for i in exp_numbers], rotation=90)
    
    # Mark the min distances
    for data in experiments_data.values():
        exp_num = data['experiment_number']
        min_distance_value = data['min_distance_value']
        min_distance_idx = data['min_distance_idx']
        # Plot the min distance
        ax.plot(exp_num, min_distance_value, 'ro')
        
        # Annotate with waypoint index (without 'WP')
        # ax.annotate(f'{min_distance_idx}', (exp_num, min_distance_value))

        # Annotate with waypoint index, positioning the text below the data point
        ax.annotate(f'{min_distance_idx}',
                    xy=(exp_num, min_distance_value),
                    xytext=(0, -5),  # Offset text 5 points below the data point
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='top')
        
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    # Shade the area below y=0
    ax.axhspan(ymin=ax.get_ylim()[0], ymax=0, facecolor='red', alpha=0.3)
    
    # # Add legend explaining the annotations
    # Create custom legend handles
    median_line = Line2D([], [], color='blue', linewidth=2, label='Median')
    mean_line = Line2D([], [], color='red', linewidth=2, linestyle='-', label='Mean')
    min_distance_point = Line2D([], [], color='red', marker='o', linestyle='None', label='Min Distance')

    # Add the legend to your plot
    ax.legend(handles=[median_line, min_distance_point], fontsize=30)
    
    # ax.text(0.95, 0.95, 'Numbers: Waypoint Indices', transform=ax.transAxes, fontsize=15,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    plots_dir = '.' # Current directory
    
    if dlo_type is None:
        plot_file = os.path.join(plots_dir, f'scene_{scene_id}_experiments_path_analysis.png')
    else:
        plot_file = os.path.join(plots_dir, f'scene_{scene_id}_dlo_{dlo_type}_experiments_path_analysis.png')
    
    # Also add "_10_segments" to the file name if saved_scores_dir contains "10_segments"
    if '10_segments' in saved_scores_dir:
        plot_file = plot_file.replace('.png', '_10_segments.png')
    
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)
    
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.close(fig)


# ------------------------------------------------------------------------

# scene_ids = [1, 2, 3, 4]
scene_ids = [0,2,6]

# DLO Types, None for the default
# dlo_types = None
# dlo_types = [1]
dlo_types = [1,4,5]

# saved_scores_dir = './scores_i9_10885h'
# saved_scores_dir = './scores_i9_10885h_10_segments'
saved_scores_dir = './scores_mingrui_yu_real_scenes'


# If dlo_types is None, set it to [None]
if dlo_types is None:
    dlo_types = [None]

for scene_id, dlo_type in itertools.product(scene_ids, dlo_types):
    plot_scene_experiments(scene_id, dlo_type, saved_scores_dir)