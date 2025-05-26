import numpy as np
import json
import random

def create_matrix(a, b, c):
    """Creates the parameterized 5x5 symmetric stochastic matrix."""
    if not check_constraints(a, b, c):
        raise ValueError("Parameters (a, b, c) do not satisfy constraints.")

    # Explicitly ensure parameters are non-negative before using them
    a_nonneg = max(0.0, a)
    b_nonneg = max(0.0, b)
    c_nonneg = max(0.0, c)

    # Calculate terms ensuring non-negativity via max(0.0, ...)
    term1 = max(0.0, 1.0 - a_nonneg - c_nonneg) # 1 - a - c
    term2 = max(0.0, 1.0 - a_nonneg - b_nonneg) # 1 - a - b
    term3 = max(0.0, 1.0 - 2.0*b_nonneg)       # 1 - 2b

    matrix = np.array([
        [term1,      0.0,        a_nonneg, 0.0,       c_nonneg],
        [0.0,        term2,      0.0,      b_nonneg,  a_nonneg],
        [a_nonneg,   0.0,        term2,    b_nonneg,  0.0],
        [0.0,        b_nonneg,   b_nonneg, term3,     0.0],
        [c_nonneg,   a_nonneg,   0.0,      0.0,       term1]
    ], dtype=float) # Ensure float type

    # Final check for negative values due to potential floating point issues
    # Use a small tolerance for the check
    if np.any(matrix < -1e-12):
        # print(f"Warning: Clamping small negative values found in matrix for a={a}, b={b}, c={c}")
        matrix[matrix < 0] = 0.0
        
    # Optional: Verify row sums are close to 1 (due to float precision)
    # if not np.allclose(matrix.sum(axis=1), 1.0):
    #     print(f"Warning: Row sums not close to 1 for a={a}, b={b}, c={c}. Sums: {matrix.sum(axis=1)}")
        
    return matrix

def check_constraints(a, b, c):
    """Checks if parameters satisfy the non-negativity constraints."""
    # Basic non-negativity (allow for small float inaccuracies)
    if not (a >= -1e-12 and b >= -1e-12 and c >= -1e-12):
        return False
    # Specific constraints (using tolerance for float comparisons)
    if not (b <= 0.5 + 1e-9):
        return False
    if not (a + b <= 1.0 + 1e-9):
        return False
    if not (a + c <= 1.0 + 1e-9):
        return False
    return True

def generate_valid_parameters():
    """Generates random (a, b, c) that satisfy the constraints."""
    max_attempts = 1000 # Prevent infinite loops in edge cases
    for _ in range(max_attempts):
        # Generate b first: b in [0, 0.5]
        b = random.uniform(0, 0.5)
        # Generate a: a in [0, 1-b]
        a = random.uniform(0, 1.0 - b)
        # Generate c: c in [0, 1-a]
        c = random.uniform(0, 1.0 - a)

        # Double check constraints and ensure parameters are non-negative
        if check_constraints(a, b, c):
            return max(0.0, a), max(0.0, b), max(0.0, c)

    raise RuntimeError("Failed to generate valid parameters after multiple attempts.")


# --- Main execution ---
num_samples = 10000 # Number of parameter sets to generate
output_filename = 'matrix_eigenvalue_data_filtered.json'
results = []

print(f"Generating {num_samples} samples for the 5x5 symmetric stochastic matrix...")
print("Eigenvalues will be sorted (descending) and the eigenvalue ~1 will be removed.")
print("Output format: 25 matrix elements, 4 remaining eigenvalues.")

samples_generated = 0
attempts = 0
max_total_attempts = num_samples * 20 # Increase attempts limit slightly if needed

while samples_generated < num_samples and attempts < max_total_attempts:
    attempts += 1
    try:
        # Generate a valid set of parameters
        a, b, c = generate_valid_parameters()

        # Create the matrix
        matrix = create_matrix(a, b, c)

        # Compute eigenvalues using eigvalsh (for symmetric matrices, returns real eigenvalues)
        eigenvalues = np.linalg.eigvalsh(matrix)

        # Sort eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1] # Sort ascending and reverse

        # Filter out the eigenvalue(s) close to 1
        tolerance = 1e-9 # Tolerance for floating point comparison with 1.0
        filtered_eigenvalues = [ev for ev in sorted_eigenvalues if not np.isclose(ev, 1.0, atol=tolerance)]

        # Format for JSON
        matrix_flat_list = matrix.flatten().tolist()
        # Convert the filtered list (already floats) to list for JSON
        eigenvalues_list = list(filtered_eigenvalues) 

        # Check expected lengths after filtering (should be 4 eigenvalues left)
        if len(matrix_flat_list) != 25:
             print(f"Warning: Unexpected matrix dimension ({len(matrix_flat_list)}) for sample {samples_generated+1}. Skipping.")
             continue
        if len(eigenvalues_list) != 4:
             # This might happen if eigenvalue 1 had multiplicity > 1 or if float issues occurred
             print(f"Warning: Unexpected number of eigenvalues ({len(eigenvalues_list)}) after filtering for sample {samples_generated+1}. Expected 4. Values: {sorted_eigenvalues}. Skipping.")
             continue

        # Store the record
        record = {
            "matrix": matrix_flat_list,
            "eigenvalues": eigenvalues_list # Store the filtered & sorted list
        }
        results.append(record)
        samples_generated += 1

        # Print progress update periodically
        if samples_generated % (num_samples // 10 if num_samples >= 10 else 1) == 0:
             print(f"Generated {samples_generated}/{num_samples} samples...")

    except ValueError as e:
        # Raised by create_matrix if check_constraints fails internally
        # print(f"Skipping sample due to ValueError: {e}") # Less verbose logging
        pass
    except np.linalg.LinAlgError as e:
        # Error during eigenvalue computation
        print(f"Skipping sample due to LinAlgError: {e}")
    except RuntimeError as e:
        # Raised by generate_valid_parameters if it fails
        print(f"Error: {e}")
        break # Stop execution


# Final check if enough samples were generated
if samples_generated < num_samples:
    print(f"\nWarning: Only generated {samples_generated} out of {num_samples} requested samples.")

# Write results to JSON file
if results:
    try:
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nSuccessfully saved {len(results)} records to {output_filename}")
    except IOError as e:
        print(f"\nError writing to file {output_filename}: {e}")
else:
    print("\nNo valid samples were generated to save.")