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

def _generate_code_from_expr(args):
    """
    Worker function that takes a pre-computed symbolic S_k expression
    and generates the final Python code for it and its Jacobian.
    """
    k, n, sk_expr, variables, var_names = args
    print(f"--- [PID:{os.getpid()}] Starting code generation for S_{k} (n={n}) ---")
    try:
        sk_code, sk_name, jac_code, jac_name = generate_sk_jacobian_code(
            k, n, sk_expr, variables, var_names
        )
        print(f"--- [PID:{os.getpid()}] Finished code generation for S_{k} (n={n}) ---")
        return {
            'k': k, 
            'sk_name': sk_name, 
            'sk_code': sk_code,
            'jac_name': jac_name, 
            'jac_code': jac_code
        }
    except Exception as e:
        print(f"!!! [PID:{os.getpid()}] ERROR during code generation for S_{k}: {e} !!!")
        return None

# ==============================================================================
# CORE SYMBOLIC AND CODE GENERATION LOGIC
# ==============================================================================

def calculate_all_sk_symbolically(M, n):
    """
    Calculates all S_k (sums of principal k-minors) for k=1..n using
    the highly efficient Newton's Sums (Faddeev-LeVerrier) algorithm.

    This avoids brute-force determinant calculation and prevents expression swell.
    """
    print("Calculating all S_k expressions using Newton's Sums...")
    start_time = time.time()
    
    # Use Newton's Sums. e_k are the elementary symmetric polynomials, which are S_k.
    # p_k are the power sums, tr(M^k).
    e = [sympy.sympify(1)] # e_0 = 1
    
    # Pre-calculate power sums p_k = tr(M^k)
    p = [n] # p_0 = tr(I) = n
    M_power_k = M
    for k in range(1, n + 1):
        p.append(sympy.trace(M_power_k))
        if k < n:
            M_power_k = M_power_k * M # Calculate next power
            
    # Apply recursive formula: k*e_k = sum_{i=1 to k} (-1)^(i-1) * e_{k-i} * p_i
    for k in range(1, n + 1):
        s = sympy.sympify(0)
        for i in range(1, k + 1):
            s += ((-1)**(i - 1)) * e[k - i] * p[i]
        e_k = s / k
        # It's often faster to expand now to keep expressions in a canonical form
        e.append(sympy.expand(e_k))

    end_time = time.time()
    print(f"Finished all symbolic S_k calculations in {end_time - start_time:.2f} seconds.")
    # Return S_1, S_2, ..., S_n
    return e[1:]

def format_expr_str(expr):
    """Replaces sympy functions with math/numpy equivalents for generated code."""
    s = str(expr).replace("sqrt", "math.sqrt").replace("Abs", "abs")
    return s

def build_matrix(matrix_type, n):
    # This function remains unchanged
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

def generate_sk_jacobian_code(k, n, sk_expr, variables, var_names):
    """
    Generates Python code for a given S_k expression and its Jacobian.
    """
    func_base_name = f"S{k}_n{n}"
    sk_func_name = f"calculate_{func_base_name}"
    jac_func_name = f"calculate_{func_base_name}_jacobian"

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
@numba.jit(nopython=True, fastmath=True, cache=True)
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
    print(f"    [k={k}] Calculating {len(variables)} partial derivatives...")
    # Differentiating the (already simplified) S_k expressions is much faster
    gradient_list = [sk_expr.diff(var) for var in variables]
    
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

    output_filename = f"{matrix_type}_symbolic_minors_n{n}.py"

    print(f"Starting code generation for N = {n} and matrix type = {matrix_type}")
    print(f"Output file: {output_filename}")
    print("-" * 30)

    variables, var_names, M = build_matrix(matrix_type, n)
    
    # 1. Calculate all S_k expressions efficiently in the main process
    all_sk_expressions = calculate_all_sk_symbolically(M, n)
    
    # 2. Parallelize the code generation for each S_k expression
    print("-" * 30)
    print(f"Executing code generation for all {len(all_sk_expressions)} S_k expressions in parallel...")
    total_start_time = time.time()
    
    map_args = [(k + 1, n, sk_expr, variables, var_names) for k, sk_expr in enumerate(all_sk_expressions)]
    
    with pp.ProcessPool() as pool:
        generated_blocks_unordered = pool.map(_generate_code_from_expr, map_args)

    # Filter out any failed jobs and sort by 'k'
    generated_blocks = [b for b in generated_blocks_unordered if b is not None]
    if generated_blocks:
        generated_blocks.sort(key=lambda x: x['k'])

    total_end_time = time.time()
    print("-" * 30)
    print(f"Total parallel code generation finished in {total_end_time - total_start_time:.2f} seconds.")

    # 3. Write results to file (same as before)
    if generated_blocks:
        print(f"Writing {len(generated_blocks)} generated function blocks to {output_filename}...")
        package_dir = os.path.dirname(__file__)
        output_filepath = os.path.join(package_dir, output_filename)

        with open(output_filepath, "w") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write(f"# Symbolic Functions for N = {n} (matrix_type='{matrix_type}')\n")
            f.write("# Generated by symbolic_minors_generator.py using Newton's Sums\n")
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

        # Update __init__.py (same as before)
        print("-" * 30)
        # ... (rest of the __init__.py update logic is unchanged) ...

    else: 
        print("No functions were generated. Check for errors in the logs above.")

    print("-" * 30); print("Script finished.")