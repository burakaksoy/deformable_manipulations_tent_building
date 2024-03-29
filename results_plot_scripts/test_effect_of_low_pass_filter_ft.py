import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.signal import firwin, lfilter
import collections
import time

# Load your CSV file
df = pd.read_csv('ft_test.csv')

# Column titles
force_torque_numbers = ['34', '5']
data_types = ['force', 'torque']
axes = ['x', 'y', 'z']
# ----------------------------------------------------------------------------------------------- #
# Simple Low-Pass Filter application

t = 0.9
start_time_simple = time.time()
for number in force_torque_numbers:
    for data_type in data_types:
        for axis in axes:
            column = f'{number}{data_type}{axis}'
            filtered_values = []
            prev_value = 0
            for current_value in df[column]:
                filtered_value = t * prev_value + (1 - t) * current_value
                filtered_values.append(filtered_value)
                prev_value = filtered_value
            df[column + '_simple_filtered'] = filtered_values
end_time_simple = time.time()

# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
# Layered Simple Low-Pass Filter application

t = 0.9
t2 = 0.9
start_time_simple2 = time.time()
for number in force_torque_numbers:
    for data_type in data_types:
        for axis in axes:
            column = f'{number}{data_type}{axis}'
            filtered_values = []
            prev_value_1 = 0
            prev_value_2 = 0
            for current_value in df[column]:
                # First layer of filtering
                filtered_value_1 = t * prev_value_1 + (1 - t) * current_value
                prev_value_1 = filtered_value_1

                # Second layer of filtering applied immediately to the result of the first
                filtered_value_2 = t2 * prev_value_2 + (1 - t2) * filtered_value_1
                filtered_values.append(filtered_value_2)
                prev_value_2 = filtered_value_2
            df[column + '_simple_filtered2'] = filtered_values
end_time_simple2 = time.time()

# ----------------------------------------------------------------------------------------------- #
# IIR Low-Pass Filter setup
fs = 50  # Define your sampling rate, Hz
cutoff = 4.0  # Define your cutoff frequency, Hz
b, a = scipy.signal.iirfilter(4, Wn=cutoff, fs=fs, btype='low', ftype='butter')
print(b, a)

start_time_iir = time.time()
# IIR Low-Pass Filter application using lfilter for faster processing
for number in force_torque_numbers:
    for data_type in data_types:
        for axis in axes:
            column = f'{number}{data_type}{axis}'
            df[column + '_iir_filtered'] = lfilter(b, a, df[column])
end_time_iir = time.time()

# ----------------------------------------------------------------------------------------------- #
# FIR Low-Pass Filter
numtaps = 101  # Number of filter taps (adjust based on your requirements)
cutoff = 2.0  # Define your cutoff frequency, Hz

# Generate FIR filter coefficients
fir_coeff = firwin(numtaps, cutoff, window='hamming', fs=fs)

start_time_fir = time.time()
for number in force_torque_numbers:
    for data_type in data_types:
        for axis in axes:
            column = f'{number}{data_type}{axis}'
            df[column + '_fir_filtered'] = lfilter(fir_coeff, 1.0, df[column])
end_time_fir = time.time()


# ----------------------------------------------------------------------------------------------- #

# Print timing results
print(f"Time taken for simple low-pass filter: {end_time_simple - start_time_simple} seconds")
print(f"Time taken for simple low-pass filter 2: {end_time_simple2 - start_time_simple2} seconds")
print(f"Time taken for IIR low-pass filter: {end_time_iir - start_time_iir} seconds")
print(f"Time taken for FIR low-pass filter: {end_time_fir - start_time_fir} seconds")


# ----------------------------------------------------------------------------------------------- #
# Plotting

# Define colors for x, y, z
colors = ['red', 'green', 'blue']

for number in force_torque_numbers:
    for data_type in data_types:
        plt.figure(figsize=(10, 6))
        # for axis in axes:
        for axis, color in zip(axes, colors):
            column = f'{number}{data_type}{axis}'

            # plt.plot(df['time'], df[column], label=f'Original {axis.upper()}', color=color, alpha=0.2, lw=2)
            plt.plot(df['time'], df[column + '_simple_filtered'], label=f'Simple Filtered {axis.upper()}', color=color, alpha=0.2, lw=1)
            plt.plot(df['time'], df[column + '_simple_filtered2'], label=f'Simple Filtered 2 {axis.upper()}', color=color, alpha=1.0, lw=1)
            # plt.plot(df['time'], df[column + '_iir_filtered'], label=f'IIR Filtered {axis.upper()}', color=color, alpha=1.0, lw=1)
            # plt.plot(df['time'], df[column + '_fir_filtered'], label=f'FIR Filtered {axis.upper()}', color=color, alpha=0.7, lw=2)

        plt.title(f'Comparison of {data_type.capitalize()} for {number}')
        plt.xlabel('Time')
        plt.ylabel(f'{data_type.capitalize()} ({number})')
        plt.legend()
        plt.show()