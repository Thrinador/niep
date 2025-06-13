import numpy as np
from scipy.optimize import check_grad

# IMPORTANT: Change this import to match the module you are working with
# For example, if you are testing S3 for a 4x4 sniep matrix:
from lib.sniep_symbolic_minors_n6 import calculate_S3_n6_value_and_jac

# --- Define the function to test ---
# Replace this with the actual function handle that is failing
# For example, if -S3 is failing, you would create the negated wrapper here too.
test_function = calculate_S3_n6_value_and_jac

# --- Helper functions to separate value and jacobian for the checker ---
def value_func(x):
    """Returns just the value from the combined function."""
    return test_function(x)[0]

def jac_func(x):
    """Returns just the jacobian from the combined function."""
    return test_function(x)[1]

# --- Run the check ---
# Set the correct number of variables for your matrix
# For n=4 sniep: num_vars = comb(4,2) = 6
num_vars = 15
test_point = np.random.rand(num_vars) # Check at a random point

# This returns the numerical difference. It should be very small.
error = check_grad(value_func, jac_func, test_point)

print(f"Jacobian verification error: {error}")
if error < 1e-6:
    print("SUCCESS: The Jacobian appears to be mathematically correct.")
else:
    print("ERROR: The Jacobian is likely incorrect. The error is large.")