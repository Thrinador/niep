import numpy as np
import plotly.graph_objects as go
import time
# Optional: for progress bar (install with: pip install tqdm)
# from tqdm import tqdm 

# --- Matrix Definition (Third User Matrix) ---

def define_matrix(a, b, c):
    """
    Defines the third 5x5 symmetric matrix provided by the user.
    """
    tol=1e-9 # Tolerance for checking constraints during matrix creation

    # These checks ensure that the matrix construction itself is based on valid params
    # The loops in generate_data_dynamic_walk are primary enforcers.
    if not (a >= -tol and b >= -tol and c >= -tol and
            a + c <= 1 + tol and
            a + b <= 1 + tol and
            b <= 0.5 + tol):
        # This case should ideally not be hit if loops are correct, but as a safeguard:
        raise ValueError(f"Parameters a={a}, b={b}, c={c} violate expected constraints for matrix construction.")

    M = np.array([
        [1-a-c, c    , 0    , a    , 0    ],
        [c    , 1-a-c, a    , 0    , 0    ],
        [0    , a    , 1-a-b, 0    , b    ],
        [a    , 0    , 0    , 1-a-b, b    ],
        [0    , 0    , b    , b    , 1-2*b]
    ])

    # Check for small negative entries due to precision
    if np.any(M < -tol):
         # print(f"Warning: Small negative entry detected for a={a:.4f},b={b:.4f},c={c:.4f}. Clamping to zero.")
         M[M < 0] = 0 # Clamp small negatives
         
    return M

# --- Validity Check (Simplified as constraints are handled by loops) ---

def is_parameters_valid(a, b, c):
    """
    Specific constraints are handled by the dynamic loops in the walk function.
    Return True unless there are *additional* complex checks needed.
    """
    return True 

# --- Eigenvalue Calculation (Same as before) ---

def get_sorted_eigenvalues(a, b, c):
    """
    Calculates eigenvalues for the defined matrix M(a, b, c) 
    using eigvalsh (assumes symmetry) and sorts them descendingly.
    Returns all 5 sorted eigenvalues or NaNs on error.
    """
    try:
        M = define_matrix(a, b, c)
        eigenvalues = np.linalg.eigvalsh(M)
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        return sorted_eigenvalues
    except Exception as e:
        # print(f"Error calculating eigenvalues for a={a:.3f}, b={b:.3f}, c={c:.3f}: {e}")
        return np.full(5, np.nan)

# --- Parameter Generation & Calculation (Structured Walk with UPDATED Dynamic Bounds) ---

def generate_data_dynamic_walk(step_a, step_b, step_c, validity_func):
    """
    Generates valid data points using a structured grid walk tailored 
    to the constraints of the THIRD matrix:
    0<=a<=1, 0<=b<=min(0.5, 1-a), 0<=c<=1-a
    """
    print(f"Generating data via dynamic structured walk with steps da={step_a}, db={step_b}, dc={step_c}...")
    tol = 1e-9 # Tolerance for floating point range comparisons
    
    # Updated range for 'a'
    a_values = np.arange(0.0, 1.0 + tol, step_a) 
    
    valid_params = {'a': [], 'b': [], 'c': []}
    lambda_results = {'l2': [], 'l3': [], 'l4': [], 'l5': []}
    
    start_time = time.time()
    checked_count = 0
    # Rough estimate
    total_potential_points_approx = (1.0/step_a) * (0.5/step_b) * (1.0/step_c) / 2 # Roughly half due to coupled constraints
    print(f"Walking grid (approx max points: {total_potential_points_approx:.0f})...") 

    # Use tqdm for progress bar if available
    # outer_loop = tqdm(a_values, desc="Walking Grid") if 'tqdm' in globals() else a_values
    outer_loop = a_values # Use this line if tqdm is not installed/imported

    for a in outer_loop:
        # Updated range for 'b'
        b_max = min(0.5, 1.0 - a) 
        if b_max < -tol: continue 
        
        b_values = np.arange(0.0, b_max + tol, step_b)
        for b in b_values:
            # Updated range for 'c'
            c_max = 1.0 - a 
            if c_max < -tol: continue 

            c_values = np.arange(0.0, c_max + tol, step_c)
            for c in c_values:
                checked_count += 1
                # Optional: Print progress manually
                # if checked_count % 20000 == 0: # Adjust frequency
                #    print(f"  Checked {checked_count} potential points...")

                # 1. Check optional user validity function (can be just True)
                if validity_func(a, b, c):
                    # 2. Calculate eigenvalues (includes matrix creation check)
                    eigenvalues = get_sorted_eigenvalues(a, b, c)
                    
                    # 3. Check if calculation succeeded
                    if not np.any(np.isnan(eigenvalues)):
                        # Store parameters and results if valid
                        valid_params['a'].append(a)
                        valid_params['b'].append(b)
                        valid_params['c'].append(c)
                        lambda_results['l2'].append(eigenvalues[1])
                        lambda_results['l3'].append(eigenvalues[2])
                        lambda_results['l4'].append(eigenvalues[3])
                        lambda_results['l5'].append(eigenvalues[4])

    end_time = time.time()
    accepted_count = len(valid_params['a'])
    print(f"\nFinished grid walk in {end_time - start_time:.2f} seconds.")
    print(f"Found {accepted_count} valid points from {checked_count} checked grid points.")
    
    # Convert results to numpy arrays
    for k in lambda_results:
        lambda_results[k] = np.array(lambda_results[k])
        
    return valid_params, lambda_results

# --- Main Execution ---

# Define step sizes for the structured grid walk
STEP_A = 0.01
STEP_B = 0.01
STEP_C = 0.01

print("--- Starting Point Cloud Plotter for Third Matrix (Structured Walk) ---")

# 1. Generate Parameters and Calculate Eigenvalues using Dynamic Grid Walk
valid_parameters, eigenvalue_data = generate_data_dynamic_walk(
    STEP_A, STEP_B, STEP_C, is_parameters_valid
)

num_plot_points = len(eigenvalue_data['l2'])

# 2. Plotting with Plotly
if num_plot_points == 0:
    print("\nNo valid data points generated to plot. Exiting.")
else:
    print(f"\nGenerating Plotly scatter plot for {num_plot_points} points...")
    
    lambda2_np = eigenvalue_data['l2']
    lambda3_np = eigenvalue_data['l3']
    lambda4_np = eigenvalue_data['l4']
    lambda5_np = eigenvalue_data['l5'] # For coloring

    fig = go.Figure(data=[go.Scatter3d(
        x=lambda2_np,
        y=lambda3_np,
        z=lambda4_np,
        mode='markers',
        marker=dict(
            size=3,                
            color=lambda5_np,      # Set color to lambda5 values
            colorscale='Viridis',  
            colorbar=dict(         # Color bar definition
                title='$\\lambda_5$ Value', 
                thickness=20
            ),
            opacity=0.7            
        ),
        hovertemplate = 
            '<b>L2</b>: %{x:.3f}<br>' +
            '<b>L3</b>: %{y:.3f}<br>' +
            '<b>L4</b>: %{z:.3f}<br>' +
            '<b>L5</b>: %{marker.color:.3f}<extra></extra>' 
    )])

    # Update plot layout
    fig.update_layout(
        title=f'Eigenvalues $(\\lambda_2, \\lambda_3, \\lambda_4)$ Colored by $\\lambda_5$ (Matrix 3, Steps: {STEP_A},{STEP_B},{STEP_C})',
        scene=dict(
            xaxis_title='$\\lambda_2$', 
            yaxis_title='$\\lambda_3$', 
            zaxis_title='$\\lambda_4$', 
            # Set ranges based on actual calculated data
            xaxis=dict(backgroundcolor='rgb(230, 230, 230)', range=[np.min(lambda2_np), np.max(lambda2_np)]),
            yaxis=dict(backgroundcolor='rgb(230, 230, 230)', range=[np.min(lambda3_np), np.max(lambda3_np)]),
            zaxis=dict(backgroundcolor='rgb(230, 230, 230)', range=[np.min(lambda4_np), np.max(lambda4_np)]),
            aspectratio=dict(x=1, y=1, z=1) 
        ),
        margin=dict(l=0, r=0, b=0, t=50) 
    )

    # Save plot to HTML file
    output_filename = "matrix3_eigenvalue_plot.html"
    print(f"Saving plot to {output_filename}...")
    fig.write_html(output_filename)
    print(f"Plot saved. Please open {output_filename} in a web browser.")
    
    # fig.show() # Removed as per previous request

    print("--- Plot generation finished ---")