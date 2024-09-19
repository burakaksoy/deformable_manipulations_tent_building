import os
import pandas as pd
import numpy as np
import pickle
import re
from scipy.stats import pearsonr, spearmanr

# Define scenes and scenarios
scenes = [1, 2, 3, 4]
scenarios = ['Auto', '10_segments']

# Define directories for each scenario
scenario_dirs = {
    'Auto': {
        'csv_saved_scores_dir': '../tesseract_planner/generated_plans_i9_10885h',
        'pickle_saved_scores_dir': './scores_i9_10885h'
    },
    '10_segments': {
        'csv_saved_scores_dir': '../tesseract_planner/generated_plans_i9_10885h_10_segments',
        'pickle_saved_scores_dir': './scores_i9_10885h_10_segments'
    }
}

# Mapping from metric internal names to desired titles
metric_titles = {
    'smoothed_peak_err': 'Max error',
    'smoothed_avr_err': 'Mean error',
    'median_smoothed_err': 'Median Error',
    'peak_err_change_on_smoothed': 'Max Error Increase',
    'avr_err_change_on_smoothed': 'Mean Error Change',
    'median_err_change_on_smoothed': 'Median Error Change',
    'min_distance_value': 'Min Distances',
    'mean_min_distances': 'Mean Min Distances',
    'median_min_distances': 'Median Min Distances'
}

# Function to load execution min distances from CSV
def load_execution_min_distances(csv_saved_scores_dir, scene_id):
    csv_file = os.path.join(csv_saved_scores_dir, f"scene_{scene_id}", f"scene_{scene_id}_experiment_execution_results.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return None
    # Load the data
    df = pd.read_csv(csv_file, na_values=['Nan', 'NaN'])
    # Remove experiments where all columns except experiment_id are NaN
    df = df.dropna(subset=[col for col in df.columns if col != 'experiment_id'], how='all')
    # Convert min_distance from meters to mm
    df['min_distance'] = df['min_distance'] * 1000  # Convert to mm
    # Remove rows where min_distance or success is NaN
    df_clean = df.dropna(subset=['min_distance', 'success'])
    # Ensure 'success' is numeric
    df_clean['success'] = pd.to_numeric(df_clean['success'], errors='coerce').astype(int)
    # Set 'experiment_id' as index
    df_clean.set_index('experiment_id', inplace=True)
    return df_clean[['min_distance', 'success']]

# Function to load error metrics from pickle files
def load_error_metrics(pickle_saved_scores_dir, scene_id):
    experiments_data = {}  # Key: experiment_number, value: data
    scene_dir = os.path.join(pickle_saved_scores_dir, f'scene_{scene_id}')
    scores_dir = os.path.join(scene_dir, 'scores')
    if not os.path.exists(scores_dir):
        print(f"Scores directory {scores_dir} does not exist.")
        return None
    # Get list of pickle files
    pickle_files = [f for f in os.listdir(scores_dir) if f.endswith('_achievability_scores.pkl')]
    for pkl_file in pickle_files:
        # Extract experiment_number from filename
        match = re.match(f'scene_{scene_id}_experiment_(\\d+)_achievability_scores.pkl', pkl_file)
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

                # Compute median values
                median_smoothed_err = np.median(smoothed_errs)
                median_err_change_on_smoothed = np.median(err_changes_on_smoothed)
                mean_min_distances = np.mean(min_distances)
                median_min_distances = np.median(min_distances)

                # Store data
                experiments_data[experiment_number] = {
                    'experiment_number': experiment_number,
                    'smoothed_peak_err': smoothed_peak_err,
                    'smoothed_avr_err': smoothed_avr_err,
                    'median_smoothed_err': median_smoothed_err,
                    'peak_err_change_on_smoothed': peak_err_change_on_smoothed,
                    'avr_err_change_on_smoothed': avr_err_change_on_smoothed,
                    'median_err_change_on_smoothed': median_err_change_on_smoothed,
                    'min_distance_value': min_distance_value,
                    'mean_min_distances': mean_min_distances,
                    'median_min_distances': median_min_distances
                }
            else:
                print(f"Scores in {pkl_file} does not have expected length 17.")
        else:
            print(f"Filename {pkl_file} does not match expected pattern.")
    # Create DataFrame from experiments_data
    df_metrics = pd.DataFrame.from_dict(experiments_data, orient='index')
    df_metrics.set_index('experiment_number', inplace=True)
    return df_metrics

# For each scene and scenario, process the data
for scene_id in scenes:
    for scenario in scenarios:
        print(f"\nProcessing Scene {scene_id}, Scenario {scenario}")
        csv_saved_scores_dir = scenario_dirs[scenario]['csv_saved_scores_dir']
        pickle_saved_scores_dir = scenario_dirs[scenario]['pickle_saved_scores_dir']

        # Load execution min distances
        df_min_distance = load_execution_min_distances(csv_saved_scores_dir, scene_id)
        if df_min_distance is None:
            continue

        # Load error metrics
        df_metrics = load_error_metrics(pickle_saved_scores_dir, scene_id)
        if df_metrics is None:
            continue

        # Merge data on experiment_id / experiment_number
        df_min_distance.index = df_min_distance.index.astype(int)
        df_metrics.index = df_metrics.index.astype(int)
        df = df_min_distance.join(df_metrics, how='inner')
        if df.empty:
            print(f"No matching experiments for Scene {scene_id}, Scenario {scenario}")
            continue

        # Compute correlation coefficients
        target_var = 'min_distance'  # Execution min distance

        # Metrics to compute correlation with
        metrics = ['smoothed_peak_err', 'smoothed_avr_err', 'median_smoothed_err',
                   'peak_err_change_on_smoothed', 'avr_err_change_on_smoothed', 'median_err_change_on_smoothed',
                   'min_distance_value', 'mean_min_distances', 'median_min_distances']

        # For each metric, compute Pearson and Spearman correlation with min_distance
        results = []
        for metric in metrics:
            if metric in df.columns:
                x = df[target_var]
                y = df[metric]
                # Remove NaNs and Infs
                mask = (~x.isna()) & (~y.isna()) & np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if len(x) > 1:
                    # Check if x or y is constant
                    if x.nunique() < 2 or y.nunique() < 2:
                        print(f"One of the variables is constant. Cannot compute correlation between {target_var} and {metric}.")
                        results.append({
                            'Metric': metric,
                            'Pearson Correlation': np.nan,
                            'Pearson p-value': np.nan,
                            'Spearman Correlation': np.nan,
                            'Spearman p-value': np.nan
                        })
                        continue
                    try:
                        pearson_corr, pearson_p = pearsonr(x, y)
                        spearman_corr, spearman_p = spearmanr(x, y)
                        results.append({
                            'Metric': metric,
                            'Pearson Correlation': pearson_corr,
                            'Pearson p-value': pearson_p,
                            'Spearman Correlation': spearman_corr,
                            'Spearman p-value': spearman_p
                        })
                    except Exception as e:
                        print(f"Error computing correlation between {target_var} and {metric}: {e}")
                        results.append({
                            'Metric': metric,
                            'Pearson Correlation': np.nan,
                            'Pearson p-value': np.nan,
                            'Spearman Correlation': np.nan,
                            'Spearman p-value': np.nan
                        })
                else:
                    print(f"Not enough data to compute correlation between {target_var} and {metric}")
                    results.append({
                        'Metric': metric,
                        'Pearson Correlation': np.nan,
                        'Pearson p-value': np.nan,
                        'Spearman Correlation': np.nan,
                        'Spearman p-value': np.nan
                    })
            else:
                print(f"Metric {metric} not found in data.")

        # Output the results in original format
        if results:
            results_df = pd.DataFrame(results)
            print(f"Correlation results for Scene {scene_id}, Scenario {scenario}:")
            print(results_df)
            print("\n")
        else:
            print(f"No results for Scene {scene_id}, Scenario {scenario}")

        # Now, output the results with metric titles and formatted numbers
        if results:
            # Map metrics to titles
            results_df_formatted = results_df.copy()
            results_df_formatted['Metric'] = results_df_formatted['Metric'].map(metric_titles).fillna(results_df_formatted['Metric'])
            # Round correlation values to two decimal places
            results_df_formatted[['Pearson Correlation', 'Pearson p-value',
                                  'Spearman Correlation', 'Spearman p-value']] = results_df_formatted[[
                'Pearson Correlation', 'Pearson p-value',
                'Spearman Correlation', 'Spearman p-value']].round(2)
            # Print formatted results
            print("Formatted Correlation Results:")
            print(results_df_formatted.to_string(index=False))
            print("\nCSV Format:")
            # Output in CSV format
            csv_output = results_df_formatted.to_csv(index=False)
            print(csv_output)
