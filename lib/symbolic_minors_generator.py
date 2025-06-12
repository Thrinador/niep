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
# This is defined at the top level to ensure it can be "pickled" by pathos.
# ==============================================================================

def _run_single_k_generation(k_args):
    """
    Wrapper function to generate Sk and its Jacobian for a single k value.
    This function is mapped across a pool of processes.
    """
    k, variables, var_names, M, n = k_args
    print(f"--- [PID:{os.getpid()}] Starting generation for k={k} ---")
    try:
        sk_code, sk_name, jac_code, jac_name = generate_sk_jacobian_code(
            variables, var_names, M, n, k
        )
        print(f"--- [PID:{os.getpid()}] Finished generation for k={k} ---")
        return {
            'k': k, 
            'sk_name': sk_name, 
            'sk_code': sk_code,
            'jac_name': jac_name, 
            'jac_code': jac_code
        }
    except Exception as e:
        print(f"!!! [PID:{os.getpid()}] ERROR during generation for k={k}: {e} !!!")
        return None


def format_expr_str(expr):
    """Replaces sympy functions with math/numpy equivalents for generated code."""
    s = str(expr).replace("sqrt", "math.sqrt").replace("Abs", "abs")
    return s

def build_matrix(matrix_type, n):
    variables = []
    variable_map = {} 
    var_names = []
    M = sympy.zeros(n, n)

    if matrix_type == 'niep':
        for i in range(n):
            for j in range(n):
                if i == j: continue
                var_name = f'x_{i}_{j}'
                sym = sympy.symbols(var_name)
                variables.append(sym)
                var_names.append(var_name)
                variable_map[(i, j)] = sym

        for i in range(n):
            for j in range(n):
                if i==j: continue
                val = variable_map[(i, j)]
                M[i, j] = val

        for i in range(n):
            off_diag_sum = sympy.sympify(0); 
            [off_diag_sum := off_diag_sum + M[i,c] for c in range(n) if i!=c]
            diag_val = 1 - off_diag_sum
            diag_val=sympy.simplify(diag_val)
            M[i, i] = diag_val

    elif matrix_type == 'sniep':
        for i in range(n):
            for j in range(i + 1, n):
                var_name = f'x_{i}_{j}'
                sym = sympy.symbols(var_name)
                variables.append(sym)
                var_names.append(var_name)
                variable_map[(i, j)] = sym

        for i in range(n):
            for j in range(i + 1, n): 
                val = variable_map[(i, j)]
                M[i, j] = val
                M[j, i] = val

        for i in range(n):
            off_diag_sum = sympy.sympify(0); 
            [off_diag_sum := off_diag_sum + M[i,c] for c in range(n) if i!=c]
            diag_val = 1 - off_diag_sum
            diag_val=sympy.simplify(diag_val)
            M[i, i] = diag_val

    elif matrix_type == 'sub_sniep':
        for i in range(n):
            for j in range(i, n):
                var_name = f'x_{i}_{j}'
                sym = sympy.symbols(var_name)
                variables.append(sym)
                var_names.append(var_name)
                variable_map[(i, j)] = sym

        for i in range(n):
            M[i,i] = variable_map[(i, i)]
            for j in range(i + 1, n): 
                M[i, j] = variable_map[(i, j)]
                M[j, i] = variable_map[(i, j)]
    
    print("Symbolic matrix constructed.")
    return variables, var_names, M


# --- Main Generation Function ---
def generate_sk_jacobian_code(variables, var_names, M, n, k):
    """
    Generates Python code strings for S_k and its Jacobian.
    This function runs sequentially within a single worker process.
    """
    start_time_gen = time.time()
    func_base_name = f"S{k}_n{n}"
    sk_func_name = f"calculate_{func_base_name}"
    jac_func_name = f"calculate_{func_base_name}_jacobian"

    # --- Sequential S_k Calculation ---
    indices = range(n)
    combinations = list(itertools.combinations(indices, k))
    total_minors = len(combinations)
    
    print(f"    [k={k}] Sequentially calculating {total_minors} determinants...")

    sk_expr = sympy.sympify(0)
    for i_calc, subset_indices in enumerate(combinations):
        submatrix = M[list(subset_indices), list(subset_indices)]
        det_expr = sympy.expand(submatrix.det())
        sk_expr += det_expr

    print(f"    [k={k}] Simplifying final S_k expression...")
    sk_expr = sympy.simplify(sk_expr)
    print(f"    [k={k}] S_k expression generated.")

    # --- Shared Code Generation Components ---
    assign_lines = [f"{var_names[i]} = x_vec[{i}]" for i in range(len(variables))]
    assign_vars_str = textwrap.indent("\n".join(assign_lines), "    ")
    wrapped_var_list = textwrap.fill(', '.join(var_names), width=70, subsequent_indent='# ')
    wrapped_var_list_doc = textwrap.fill(', '.join(var_names), width=65, initial_indent='           ', subsequent_indent='           ')
    common_docstring = f"""
    This function is JIT-compiled with Numba for performance.

    Args:
        x_vec (numpy.ndarray): Input vector of length {len(variables)}
           containing the variable matrix elements in the specified order:
           {wrapped_var_list_doc}
    """

    # --- Generate S_k Function Code ---
    sk_repl, sk_red = sympy.cse(sk_expr, optimizations='basic')
    sk_final_str = format_expr_str(sk_red[0]) if sk_red else format_expr_str(sk_expr)
    sk_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in sk_repl]
    sk_cse_defs_str = textwrap.indent("\n".join(sk_cse_lines), "    ")
    sk_code = f"""
# --------------------------------------------------------------------------
# Value Function ({func_base_name})
# Input: {wrapped_var_list}
# --------------------------------------------------------------------------
@numba.jit(nopython=True)
def {sk_func_name}(x_vec):
    \"\"\"Calculates S_{k} for n={n} using generated symbolic expressions.{common_docstring}
    Returns:
        float: The value of S_{k}.
    \"\"\"
    # Assign vars
{assign_vars_str}
    # S_k CSE Defs
{sk_cse_defs_str}
    # Final S_k Calculation
    result = {sk_final_str}
    return result
"""

    # --- Generate Jacobian Function Code ---
    gradient_list = []
    print(f"    [k={k}] Calculating {len(variables)} partial derivatives for Jacobian...")
    for i_calc, var in enumerate(variables):
        deriv = sk_expr.diff(var)
        deriv = sympy.simplify(deriv)
        gradient_list.append(deriv)
    
    print(f"    [k={k}] Running CSE on Jacobian expressions...")
    jac_repl, jac_red = sympy.cse(gradient_list, optimizations='basic')
    jac_final_strs = [format_expr_str(expr) for expr in jac_red] if jac_red else [format_expr_str(g) for g in gradient_list]
    if len(jac_final_strs) != len(variables): raise ValueError("Jac expr count mismatch")
    jac_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in jac_repl]
    jac_cse_defs_str = textwrap.indent("\n".join(jac_cse_lines), "    ")
    jac_return_list_str = ",\n        ".join(jac_final_strs)
    jac_code = f"""
# --------------------------------------------------------------------------
# Jacobian Function ({func_base_name})
# Input: {wrapped_var_list}
# --------------------------------------------------------------------------
@numba.jit(nopython=True)
def {jac_func_name}(x_vec):
    \"\"\"Calculates the Jacobian of S_{k} for n={n}.{common_docstring}
    Returns:
        numpy.ndarray: The gradient vector of S_{k}.
    \"\"\"
    # Assign vars
{assign_vars_str}
    # Jacobian CSE Defs
{jac_cse_defs_str}
    # Final Gradient Calculation
    gradient = np.array([
        {jac_return_list_str}
    ])
    return gradient
"""
    end_time_gen = time.time()
    print(f"    [k={k}] Code generation finished in {end_time_gen - start_time_gen:.2f} seconds.")
    return sk_code, sk_func_name, jac_code, jac_func_name

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(parent_dir, 'config.toml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), 'config.toml')

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    n = config['global_data']['n']
    matrix_type = config['global_data']['matrix_type']

    k_options = list(range(1, n + 1))
    output_filename = f"{matrix_type}_symbolic_minors_n{n}.py"

    print(f"Starting code generation for N = {n} and matrix type = {matrix_type}")
    print(f"Target k values: {k_options}"); print(f"Output file: {output_filename}")
    print("-" * 30)

    variables, var_names, M = build_matrix(matrix_type, n)
    
    # --- Parallel Generation for all k values ---
    print(f"Executing generation for all {len(k_options)} k-values in parallel...")
    total_start_time = time.time()
    
    # Package arguments for the map function. This is a robust way to pass multiple
    # arguments to the worker function.
    map_args = [(k, variables, var_names, M, n) for k in k_options]
    
    with pp.ProcessPool() as pool:
        # Map the generation function across all k values
        generated_blocks_unordered = pool.map(_run_single_k_generation, map_args)

    # Filter out any failed jobs (which return None)
    generated_blocks = [b for b in generated_blocks_unordered if b is not None]
    
    # IMPORTANT: Sort results by 'k' to ensure file is written in a deterministic order
    if generated_blocks:
        generated_blocks.sort(key=lambda x: x['k'])

    total_end_time = time.time()
    print("-" * 30)
    print(f"Total parallel generation finished in {total_end_time - total_start_time:.2f} seconds.")

    if generated_blocks:
        print(f"Writing {len(generated_blocks)} generated function blocks to {output_filename}...")
        package_dir = os.path.dirname(__file__)
        output_filepath = os.path.join(package_dir, output_filename)

        with open(output_filepath, "w") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write(f"# Symbolic Functions for N = {n} (matrix_type='{matrix_type}')\n")
            f.write("# Generated by symbolic_minors_generator.py\n")
            f.write("# DO NOT EDIT MANUALLY\n\n")
            f.write("import math\n")
            f.write("import numpy as np\n")
            f.write("import numba\n\n")

            for block in generated_blocks:
                f.write(block['sk_code'])
                f.write("\n\n")
                if block['jac_code']: 
                    f.write(block['jac_code'])
                    f.write("\n\n")
        print(f"Successfully wrote code to {output_filepath}")

        # --- Integrated __init__.py Update Logic ---
        print("-" * 30)
        print("Updating package __init__.py to reflect generated files...")
        init_file_path = os.path.join(package_dir, '__init__.py')
        module_names_to_import = []
        glob_pattern = os.path.join(package_dir, '*_symbolic_minors_n*.py')
        for path in glob.glob(glob_pattern):
            module_name = os.path.splitext(os.path.basename(path))[0]
            module_names_to_import.append(module_name)
        
        try:
            with open(init_file_path, 'w') as f:
                f.write("# This file is auto-generated by a script. Do not edit manually.\n\n")
                f.write("# Manually-defined package members\n")
                f.write("from . import file_utils\n")
                f.write("from . import optimize_tasks\n")
                f.write("from . import eigenvalue_tasks\n")
                f.write("from . import plot_utils\n\n")
                f.write("# Auto-generated symbolic modules\n")
                for name in sorted(module_names_to_import):
                    f.write(f"from . import {name}\n")
            print(f"Successfully updated {init_file_path}")
        except IOError as e:
            print(f"ERROR: Could not write to {init_file_path}: {e}")

    else: 
        print("No functions were generated. Check for errors in the logs above.")

    print("-" * 30); print("Script finished.")