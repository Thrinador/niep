import sympy
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

def _generate_separate_codes_for_sk(args):
    """
    Worker function that generates separate code for value, jacobian, AND hessian.
    """
    k, n, sk_expr, variables, var_names = args
    process_id = os.getpid()
    print(f"--- [PID:{process_id}] Starting separate code generation for S_{k} ---")
    try:
        val_code, val_name, jac_code, jac_name, hess_code, hess_name = generate_separate_functions_code(
            k, n, sk_expr, variables, var_names
        )
        print(f"--- [PID:{process_id}] Finished code generation for S_{k} ---")
        return {
            'k': k,
            'val_code': val_code, 'val_name': val_name,
            'jac_code': jac_code, 'jac_name': jac_name,
            'hess_code': hess_code, 'hess_name': hess_name
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
    
    e = [sympy.sympify(1)]
    p = [n]
    M_power_k = M
    for k in range(1, n + 1):
        p.append(sympy.trace(M_power_k))
        if k < n:
            # Avoid re-calculating M*M*...*M each time
            M_power_k = M_power_k * M
            
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
    variables, var_names = [], []
    M = sympy.zeros(n, n)

    # Simplified logic for variable creation
    if matrix_type == 'sub_sniep':
        # Symmetric, not necessarily stochastic, variables include the diagonal
        for i in range(n):
            for j in range(i, n):
                var_name = f'x_{i}_{j}'
                sym = sympy.symbols(var_name)
                variables.append(sym)
                var_names.append(var_name)
                M[i, j] = M[j, i] = sym
    else: # niep or sniep
        # Off-diagonal variables
        is_symmetric = matrix_type == 'sniep'
        rows, cols = (range(n), range(n))
        for i in rows:
            for j in cols:
                if (is_symmetric and j <= i) or (not is_symmetric and i == j):
                    continue
                var_name = f'x_{i}_{j}'
                sym = sympy.symbols(var_name)
                variables.append(sym)
                var_names.append(var_name)
                M[i, j] = sym
                if is_symmetric:
                    M[j, i] = sym
        
        # Diagonal elements from stochastic constraint
        for i in range(n):
            off_diag_sum = sum(M[i, c] for c in range(n) if i != c)
            M[i, i] = 1 - off_diag_sum
    
    print("Symbolic matrix constructed.")
    return variables, var_names, M

def generate_separate_functions_code(k, n, sk_expr, variables, var_names):
    """
    Generates three separate Python function strings for:
    1. S_k value
    2. S_k Jacobian
    3. S_k Hessian
    """
    num_vars = len(variables)
    base_name = f"S{k}_n{n}"
    
    # --- Shared Components ---
    assign_lines = [f"{var_names[i]} = x_vec[{i}]" for i in range(num_vars)]
    assign_vars_str = textwrap.indent("\n".join(assign_lines), "    ")
    numba_decorator = "@numba.jit(nopython=True, fastmath=True, cache=True)"

    # --- 1. Value Function Generation ---
    print(f"    [k={k}] Generating Value function...")
    val_name = f"calculate_{base_name}"
    val_repl, val_red = sympy.cse([sk_expr], optimizations='basic')
    val_cse_lines = [f"{s} = {format_expr_str(e)}" for s, e in val_repl]
    val_cse_str = textwrap.indent("\n".join(val_cse_lines), "    ")
    val_final_str = format_expr_str(val_red[0])
    val_code = f"""
{numba_decorator}
def {val_name}(x_vec):
    \"\"\"Calculates the value of S_{k} for n={n}.\"\"\"
{assign_vars_str}
{val_cse_str}
    return {val_final_str}
"""

    # --- 2. Jacobian Function Generation ---
    print(f"    [k={k}] Generating Jacobian function...")
    jac_name = f"calculate_{base_name}_jacobian"
    gradient_list = [sk_expr.diff(var) for var in variables]
    jac_repl, jac_red = sympy.cse(gradient_list, optimizations='basic')
    jac_cse_lines = [f"{s} = {format_expr_str(e)}" for s, e in jac_repl]
    jac_cse_str = textwrap.indent("\n".join(jac_cse_lines), "    ")
    jac_final_list_str = ",\n        ".join(map(format_expr_str, jac_red))
    jac_code = f"""
{numba_decorator}
def {jac_name}(x_vec):
    \"\"\"Calculates the Jacobian of S_{k} for n={n}.\"\"\"
{assign_vars_str}
{jac_cse_str}
    return np.array([{jac_final_list_str}])
"""

    # --- 3. Hessian Function Generation ---
    print(f"    [k={k}] Generating Hessian function... (This may be slow)")
    hess_name = f"calculate_{base_name}_hessian"
    hessian_matrix = sympy.zeros(num_vars, num_vars)
    for i in range(num_vars):
        for j in range(i, num_vars):
            deriv = gradient_list[i].diff(variables[j])
            hessian_matrix[i, j] = deriv
            if i != j:
                hessian_matrix[j, i] = deriv

    hess_flat_list = [hessian_matrix[i, j] for i in range(num_vars) for j in range(num_vars)]
    hess_repl, hess_red = sympy.cse(hess_flat_list, optimizations='basic')
    hess_cse_lines = [f"{s} = {format_expr_str(e)}" for s, e in hess_repl]
    hess_cse_str = textwrap.indent("\n".join(hess_cse_lines), "    ")
    
    hess_code = f"""
{numba_decorator}
def {hess_name}(x_vec):
    \"\"\"Calculates the Hessian matrix of S_{k} for n={n}.\"\"\"
{assign_vars_str}
{hess_cse_str}
    hessian = np.empty(({num_vars}, {num_vars}))
"""
    for i in range(num_vars):
        for j in range(num_vars):
            idx = i * num_vars + j
            hess_code += f"    hessian[{i}, {j}] = {format_expr_str(hess_red[idx])}\n"
    hess_code += "    return hessian\n"

    return val_code, val_name, jac_code, jac_name, hess_code, hess_name


if __name__ == "__main__":
    # Set a higher recursion limit for the sympy/Numba compilation of large expressions
    try:
        new_limit = 15000
        print(f"Setting Python recursion limit to {new_limit} for large matrix generation.")
        sys.setrecursionlimit(new_limit)
    except Exception as e:
        print(f"Warning: Could not set recursion limit: {e}")

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
        generated_blocks_unordered = pool.map(_generate_separate_codes_for_sk, map_args)

    generated_blocks = [b for b in generated_blocks_unordered if b is not None]
    if generated_blocks:
        generated_blocks.sort(key=lambda x: x['k'])

    total_end_time = time.time()
    print("-" * 30)
    print(f"Total parallel code generation finished in {total_end_time - total_start_time:.2f} seconds.")

    if generated_blocks:
        print(f"Writing {len(generated_blocks) * 3} generated functions to {output_filename}...")
        package_dir = os.path.dirname(__file__) or '.'
        output_filepath = os.path.join(package_dir, output_filename)

        with open(output_filepath, "w") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write(f"# Separated Value, Jacobian, & Hessian Functions for N = {n} (matrix_type='{matrix_type}')\n")
            f.write("# Generated by symbolic_minors_generator.py\n")
            f.write("# DO NOT EDIT MANUALLY\n\n")
            f.write("import numpy as np\n")
            f.write("import numba\n\n")

            for block in generated_blocks:
                f.write(f"# --- Functions for S_{block['k']} ---\n")
                f.write(block['val_code'])
                f.write("\n\n")
                f.write(block['jac_code'])
                f.write("\n\n")
                f.write(block['hess_code'])
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