from collections import deque

class MovingAverageWithConvergenceAndTrend:
    def __init__(self, n, tolerance=1e-4, trend_tolerance=1e-3, stable_count=5):
        self.n = n
        self.tolerance = tolerance  # Convergence threshold for moving average
        self.trend_tolerance = trend_tolerance  # Tolerance for detecting trend stability
        self.stable_count = stable_count  # Number of consecutive stable readings
        self.reset()
    
    def update(self, new_value):
        if len(self.window) == self.n:
            # Remove the oldest value from the sum and adjust the net trend change
            self.sum -= self.window[0]
            self.net_change -= self.window[1] - self.window[0]  # Oldest -> second oldest change
        
        if len(self.window) > 0:
            # Adjust net change based on the new value being added
            self.net_change += new_value - self.window[-1]
        
        self.window.append(new_value)
        self.sum += new_value
        current_avg = self.sum / len(self.window)
        
        # Check for convergence and trend stability
        converged = self.check_convergence(current_avg) and self.check_trend_stability()
        
        # Update the previous average
        self.previous_avg = current_avg
        
        return current_avg, converged
    
    def check_convergence(self, current_avg):
        """Check whether the moving average is stable."""
        if self.previous_avg is None:
            return False  # Not enough data to check convergence yet
        
        # Check if the change between consecutive averages is within the tolerance
        if abs(current_avg - self.previous_avg) < self.tolerance:
            self.stable_counter += 1
        else:
            self.stable_counter = 0  # Reset if not stable
        
        # Converged if we have had enough consecutive stable readings
        return self.stable_counter >= self.stable_count
    
    def check_trend_stability(self):
        """Check whether the cumulative trend in the window is stable."""
        # The trend is stable if the net cumulative change is within the trend tolerance
        return abs(self.net_change) < self.trend_tolerance
    
    def reset(self):
        """Reset the moving average, convergence, and trend analysis state."""
        self.window = deque(maxlen=self.n)
        self.sum = 0.0
        self.net_change = 0.0  # Track the cumulative trend change
        self.stable_counter = 0
        self.previous_avg = None