import sympy
import itertools
import os
import textwrap
import time
import numpy as np
import tomli
import glob
import pathos.pools as pp

# ==============================================================================
# HELPER FUNCTION FOR PARALLELIZATION
# ==============================================================================

def _generate_combined_code_for_sk(args):
    """
    Worker function that takes a pre-computed symbolic S_k expression
    and generates a single, combined function for its value and Jacobian.
    """
    k, n, sk_expr, variables, var_names = args
    process_id = os.getpid()
    print(f"--- [PID:{process_id}] Starting combined code generation for S_{k} ---")
    try:
        combined_code, combined_func_name = generate_combined_function_code(
            k, n, sk_expr, variables, var_names
        )
        print(f"--- [PID:{process_id}] Finished code generation for S_{k} ---")
        return {
            'k': k, 
            'func_name': combined_func_name, 
            'func_code': combined_code
        }
    except Exception as e:
        print(f"!!! [PID:{process_id}] ERROR during code generation for S_{k}: {e} !!!")
        return None

# ==============================================================================
# CORE SYMBOLIC AND CODE GENERATION LOGIC
# ==============================================================================

def calculate_all_sk_symbolically(M, n):
    """
    Calculates all S_k expressions for k=1..n using Newton's Sums.
    """
    print("Calculating all S_k expressions using Newton's Sums...")
    start_time = time.time()
    
    e = [sympy.sympify(1)] # e_0 = 1
    p = [n] # p_0 = tr(I) = n
    M_power_k = M
    for k in range(1, n + 1):
        p.append(sympy.trace(M_power_k))
        if k < n:
            M_power_k = M_power_k * M # Calculate next power
            
    for k in range(1, n + 1):
        s = sympy.sympify(0)
        for i in range(1, k + 1):
            s += ((-1)**(i - 1)) * e[k - i] * p[i]
        e_k = s / k
        e.append(sympy.expand(e_k))

    end_time = time.time()
    print(f"Finished all symbolic S_k calculations in {end_time - start_time:.2f} seconds.")
    return e[1:]

def format_expr_str(expr):
    """Replaces sympy functions with math/numpy equivalents for generated code."""
    s = str(expr).replace("sqrt", "math.sqrt").replace("Abs", "abs")
    return s

def build_matrix(matrix_type, n):
    """Constructs the symbolic matrix M."""
    variables, variable_map, var_names = [], {}, []
    M = sympy.zeros(n, n)

    if matrix_type == 'sniep' or matrix_type == 'niep':
        # Define off-diagonal variables
        if matrix_type == 'sniep': # Symmetric
            for i in range(n):
                for j in range(i + 1, n):
                    var_name = f'x_{i}_{j}'
                    sym = sympy.symbols(var_name)
                    variables.append(sym)
                    var_names.append(var_name)
                    M[i, j] = M[j, i] = sym
        else: # Non-symmetric
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    var_name = f'x_{i}_{j}'
                    sym = sympy.symbols(var_name)
                    variables.append(sym)
                    var_names.append(var_name)
                    M[i, j] = sym
        
        # Define diagonal elements by stochastic constraint
        for i in range(n):
            off_diag_sum = sympy.sympify(0)
            for c in range(n):
                if i != c:
                    off_diag_sum += M[i, c]
            M[i, i] = 1 - off_diag_sum

    elif matrix_type == 'sub_sniep':
        for i in range(n):
            for j in range(i, n):
                var_name = f'x_{i}_{j}'
                sym = sympy.symbols(var_name)
                variables.append(sym)
                var_names.append(var_name)
                M[i, i] = sym if i == j else M[i,i]
                if i != j: M[i, j] = M[j, i] = sym
    
    print("Symbolic matrix constructed.")
    return variables, var_names, M

def generate_combined_function_code(k, n, sk_expr, variables, var_names):
    """
    Generates a single, combined Python function string for S_k and its Jacobian.
    """
    func_base_name = f"S{k}_n{n}"
    combined_func_name = f"calculate_{func_base_name}_value_and_jac"

    # --- Generate Jacobian and Run Combined CSE ---
    print(f"    [k={k}] Calculating {len(variables)} partial derivatives...")
    gradient_list = [sk_expr.diff(var) for var in variables]
    
    print(f"    [k={k}] Running CSE on combined value and gradient expressions...")
    # Key optimization: Pass the main expression AND the gradient list to CSE
    repl, red = sympy.cse([sk_expr] + gradient_list, optimizations='basic')

    sk_final_str = format_expr_str(red[0])
    jac_final_strs = [format_expr_str(expr) for expr in red[1:]]

    # --- Generate Code Strings ---
    assign_lines = [f"{var_names[i]} = x_vec[{i}]" for i in range(len(variables))]
    assign_vars_str = textwrap.indent("\n".join(assign_lines), "    ")
    wrapped_var_list = textwrap.fill(', '.join(var_names), width=70, subsequent_indent='# ')
    wrapped_var_list_doc = textwrap.fill(', '.join(var_names), width=65, initial_indent='           ', subsequent_indent='           ')
    
    cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in repl]
    cse_defs_str = textwrap.indent("\n".join(cse_lines), "    ")
    jac_return_list_str = ",\n        ".join(jac_final_strs)
    
    # --- Create Final Function String ---
    combined_code = f"""
# --------------------------------------------------------------------------
# Combined Value and Jacobian Function ({func_base_name})
# Generated for use with scipy.optimize.minimize(..., jac=True)
# --------------------------------------------------------------------------
@numba.jit(nopython=True, fastmath=True, cache=True)
def {combined_func_name}(x_vec):
    \"\"\"Calculates both the value and Jacobian of S_{k} for n={n}.

    Args:
        x_vec (numpy.ndarray): Input vector of length {len(variables)}
           containing the variable matrix elements in the specified order:
           {wrapped_var_list_doc}

    Returns:
        (float, numpy.ndarray): A tuple containing the S_{k} value and its gradient.
    \"\"\"
    # Assign variables from input vector
{assign_vars_str}

    # Common subexpressions for value and gradient
{cse_defs_str}

    # Final calculations
    result = {sk_final_str}
    gradient = np.array([
        {jac_return_list_str}
    ])
    
    return result, gradient
"""
    return combined_code, combined_func_name

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(__file__)) or '.'
    config_path = os.path.join(parent_dir, 'config.toml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), 'config.toml')

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    n = config['global_data']['n']
    matrix_type = config['global_data']['matrix_type']

    output_filename = f"{matrix_type}_symbolic_minors_n{n}.py"

    print(f"Starting code generation for N = {n}, matrix type = '{matrix_type}'")
    print(f"Output file: {output_filename}")
    print("-" * 30)

    variables, var_names, M = build_matrix(matrix_type, n)
    all_sk_expressions = calculate_all_sk_symbolically(M, n)
    
    print("-" * 30)
    print(f"Executing code generation for all {len(all_sk_expressions)} S_k expressions in parallel...")
    total_start_time = time.time()
    
    map_args = [(k + 1, n, sk_expr, variables, var_names) for k, sk_expr in enumerate(all_sk_expressions)]
    
    with pp.ProcessPool() as pool:
        generated_blocks_unordered = pool.map(_generate_combined_code_for_sk, map_args)

    generated_blocks = [b for b in generated_blocks_unordered if b is not None]
    if generated_blocks:
        generated_blocks.sort(key=lambda x: x['k'])

    total_end_time = time.time()
    print("-" * 30)
    print(f"Total parallel code generation finished in {total_end_time - total_start_time:.2f} seconds.")

    if generated_blocks:
        print(f"Writing {len(generated_blocks)} generated function blocks to {output_filename}...")
        package_dir = os.path.dirname(__file__) or '.'
        output_filepath = os.path.join(package_dir, output_filename)

        with open(output_filepath, "w") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write(f"# Combined Value & Jacobian Functions for N = {n} (matrix_type='{matrix_type}')\n")
            f.write("# Generated by symbolic_minors_generator.py using Newton's Sums\n")
            f.write("# Optimized for use with scipy.optimize.minimize(..., jac=True)\n")
            f.write("# DO NOT EDIT MANUALLY\n\n")
            f.write("import numpy as np\n")
            f.write("import numba\n")
            f.write("# The 'math' module is not needed as Numba recognizes standard functions\n\n")

            for block in generated_blocks:
                f.write(block['func_code'])
                f.write("\n\n")
        print(f"Successfully wrote code to {output_filepath}")
        
        # -------------------------------------------------------------------
        # ### BEGIN: INTEGRATED __init__.py UPDATE LOGIC ###
        # -------------------------------------------------------------------
        print("-" * 30)
        print("Updating package __init__.py to reflect generated files...")

        init_file_path = os.path.join(package_dir, '__init__.py')

        # Find all symbolic module files using a glob pattern
        # This ensures we find all N values, not just the one we just made
        module_names_to_import = []
        glob_pattern = os.path.join(package_dir, '*_symbolic_minors_n*.py')
        for path in glob.glob(glob_pattern):
            module_name = os.path.splitext(os.path.basename(path))[0]
            module_names_to_import.append(module_name)
        
        # You can add other manually-created files to the list if needed
        # For example: module_names_to_import.append('symbolic_minors_generator')

        try:
            with open(init_file_path, 'w') as f:
                f.write("# This file is auto-generated. Do not edit manually.\n\n")
                
                # Add any other modules that should always be part of the package API
                f.write("# Manually-defined package members\n")
                f.write("from . import file_utils\n")
                f.write("from . import optimize_tasks\n")
                f.write("from . import eigenvalue_tasks\n\n")
                f.write("from . import plot_utils\n\n")

                # Add the dynamically found modules
                f.write("# Auto-generated symbolic modules\n")
                for name in sorted(module_names_to_import):
                    f.write(f"from . import {name}\n")
            
            print(f"Successfully updated {init_file_path}")
        
        except IOError as e:
            print(f"ERROR: Could not write to {init_file_path}: {e}")
        # -----------------------------------------------------------------
        # ### END: INTEGRATED __init__.py UPDATE LOGIC ###
        # -----------------------------------------------------------------

    else: 
        print("No functions were generated. Check for errors in the logs above.")

    print("-" * 30)
    print("Script finished.")