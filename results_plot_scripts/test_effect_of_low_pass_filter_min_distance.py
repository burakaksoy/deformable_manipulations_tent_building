import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import collections
import time

# Define the LiveLFilter class for IIR filtering
class LiveLFilter:
    def __init__(self, b, a):
        self.a = a
        self.b = b
        self._xs = collections.deque([0] * len(b), maxlen=len(b))
        self._ys = collections.deque([0] * (len(a) - 1), maxlen=len(a) - 1)

    def process(self, x):
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y /= self.a[0]
        self._ys.appendleft(y)
        return y

# Load your CSV file
df = pd.read_csv('min_distance_test.csv')

# Simple Low-Pass Filter
t = 0.5
# columns_to_filter = ['minDistance0', 'minDistance1', 'minDistance2', 'minDistance3']
columns_to_filter = ['minDistance3']

start_time_simple = time.time()
for column in columns_to_filter:
    filtered_values = []
    prev_value = 0
    for current_value in df[column]:
        filtered_value = t * prev_value + (1 - t) * current_value
        filtered_values.append(filtered_value)
        prev_value = filtered_value
    df[column + '_simple_filtered'] = filtered_values
end_time_simple = time.time()

# IIR Low-Pass Filter
fs = 50  # Define your sampling rate, Hz
cutoff = 2.0  # Define your cutoff frequency, Hz
b, a = scipy.signal.iirfilter(4, Wn=cutoff, fs=fs, btype='low', ftype='butter')

start_time_iir = time.time()
for column in columns_to_filter:
    live_filter = LiveLFilter(b, a)
    df[column + '_iir_filtered'] = [live_filter.process(x) for x in df[column]]
end_time_iir = time.time()

# Print the time taken for each filter
print(f"Time taken for simple low-pass filter: {end_time_simple - start_time_simple} seconds")
print(f"Time taken for IIR low-pass filter: {end_time_iir - start_time_iir} seconds")

# Plotting
for column in columns_to_filter:
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df[column], label='Original')
    plt.plot(df['time'], df[column + '_simple_filtered'], label='Simple Filtered', linestyle='--')
    plt.plot(df['time'], df[column + '_iir_filtered'], label='IIR Filtered', linestyle='-.')
    plt.title(f'Comparison of Filters for {column}')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.legend()
    plt.show()