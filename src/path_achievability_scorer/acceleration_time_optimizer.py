import cvxpy as cp

"""
Author: Burak Aksoy

This code provides an example of how to use CVXPY to solve a simple optimization problem 
with a class-based approach. 
Class-based approach is useful when you want to solve the same optimization problem EFFICIENTLY
with different parameter values.

For more insights on the effectiveness of the class-based approach,
refer to the following link:
https://github.com/cvxpy/cvxpy/issues/762
"""

class AccelerationTimeOptimizer:
    def __init__(self):
        # Define the optimization variable
        self.x = cp.Variable()

        # Define parameters
        self.x_0 = cp.Parameter()
        self.a = cp.Parameter()
        self.b = cp.Parameter()
        self.c = cp.Parameter()
        self.d = cp.Parameter()

        # Define the constraints
        self.constraints = [
            self.x**2 + self.a*self.x <= self.b,  # Quadratic constraint
            self.x <= self.c,                    # Upper bound constraint for c
            0 < self.x,                          # Lower bound constraint for positive x
            self.x <= self.d                     # Upper bound constraint for d
        ]

        # Define the objective function (minimizing the squared distance from x_0)
        self.objective = cp.Minimize(cp.square(self.x - self.x_0))

        # Define the problem
        self.problem = cp.Problem(self.objective, self.constraints)

    def solve(self, x_0_new, a_new, b_new, c_new, d_new):
        # Update parameter values
        self.x_0.value = x_0_new
        self.a.value = a_new
        self.b.value = b_new
        self.c.value = c_new
        self.d.value = d_new

        # Solve the problem with warm start to improve efficiency
        self.problem.solve(warm_start=True)

        # Return the optimal value of x
        return self.x.value

# # Example usage:
# optimizer = ClosestXOptimizer()

# # Solve with new parameters
# x_0 = 2.0
# a = 1.5
# b = 5.0
# c = 2.5
# d = 3.5
# optimal_x = optimizer.solve(x_0, a, b, c, d)
# print(f"The optimal value of x is: {optimal_x}")