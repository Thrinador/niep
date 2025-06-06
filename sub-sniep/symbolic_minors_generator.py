import sympy
import itertools
import os
import textwrap
import time
import numpy as np

# --- Function to Format Expression Strings ---
def format_expr_str(expr):
    """Replaces sympy functions with math/numpy equivalents for generated code."""
    s = str(expr).replace("sqrt", "math.sqrt").replace("Abs", "abs")
    return s

# --- Main Generation Function ---
def generate_sk_jacobian_code(n, k):
    """
    Generates Python code strings for S_k, its Jacobian
    for a general symmetric substochastic matrix (parameterized by upper triangle).
    Includes an option for potentially slow aggressive simplification.
    """

    if k < 1 or k > n: 
        print(f"Error: k={k} out of range for n={n}."); 
        return None, None, None, None

    start_time_gen = time.time()
    # Define function names consistently
    func_base_name = f"S{k}_n{n}"
    sk_func_name = f"calculate_{func_base_name}"
    jac_func_name = f"calculate_{func_base_name}_jacobian"

    # 1. Define ALL symbols for the strict upper triangle
    variables = []
    variable_map = {} 
    var_names = []

    for i in range(n):
        for j in range(i, n):
            var_name = f'x_{i}_{j}'
            sym = sympy.symbols(var_name)
            variables.append(sym)
            var_names.append(var_name)
            variable_map[(i, j)] = sym

    num_vars = len(variables)
    print(f"Generating S_{k}, Jac for N={n} general symm. stochastic ({num_vars} variables)...")

    # Initialize return values
    sk_code = None
    jac_code = None
    sk_expr = None

    # 2. Build symbolic matrix & Calculate S_k (within a try block)
    try:
        M = sympy.zeros(n, n)
        for i in range(n):
            M[i,i] = variable_map[(i, i)]
            for j in range(i + 1, n): 
                M[i, j] = variable_map[(i, j)]
                M[j, i] = variable_map[(i, j)]
        print("... Symbolic matrix constructed.")

        sk_expr = sympy.sympify(0)
        indices = range(n)
        total_minors = sympy.binomial(n, k)

        for i_calc, subset_indices in enumerate(itertools.combinations(indices, k)):
            submatrix = M[list(subset_indices), list(subset_indices)] 
            det_expr = sympy.expand(submatrix.det())
            sk_expr += det_expr
            if (i_calc+1)%max(1,total_minors//10)==0 or (i_calc+1)==total_minors: 
                print(f"    S_k Det {i_calc+1}/{total_minors}...")
            sk_expr=sympy.simplify(sk_expr); 
        print("... Symbolic S_k calculation complete.")

    except Exception as e: 
        print(f"ERROR S_k calc: {e}")
        traceback.print_exc(); 
        return None, None, None, None

    # --- Shared Code Generation Components ---
    assign_lines = [f"{var_names[i]} = x_vec[{i}]" for i in range(num_vars)]
    assign_vars_str = textwrap.indent("\n".join(assign_lines), "    ")
    wrapped_var_list = textwrap.fill(', '.join(var_names), width=70, subsequent_indent='# ')
    wrapped_var_list_doc = textwrap.fill(', '.join(var_names), width=65, initial_indent='           ', subsequent_indent='           ')
    common_docstring = f"""
    Args:
        x_vec (numpy array or list/tuple): Input vector of length {num_vars}
           containing the strict upper triangle elements in row-wise order:
           {wrapped_var_list_doc}
           Diagonal elements are determined by stochastic constraint.
    """

    # --- Generate S_k Function Code ---
    try: # Try S_k code generation
        sk_repl, sk_red = sympy.cse(sk_expr, optimizations='basic')
        sk_final_str = format_expr_str(sk_red[0]) if sk_red else format_expr_str(sk_expr)
        sk_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in sk_repl]
        sk_cse_defs_str = textwrap.indent("\n".join(sk_cse_lines), "    ")
        sk_code = f"""
# --------------------------------------------------------------------------
# Value Function ({func_base_name})
# Input: {wrapped_var_list}
# --------------------------------------------------------------------------
def {sk_func_name}(x_vec):
    \"\"\"Calculates S_{k} for n={n} general symm. stochastic matrix.{common_docstring}
    Returns: float: S_{k} value.
    \"\"\"
    # Assign vars
{assign_vars_str}
    # S_k CSE Defs
{sk_cse_defs_str}
    # Final S_k Calculation
    result = {sk_final_str}
    return result
"""
        print(f"... Code for {sk_func_name} ready.")
    except Exception as e: print(f"Error S_k CodeGen: {e}"); sk_code = None

    # --- Generate Jacobian Function Code ---
    if sk_code and sk_expr is not None:
        gradient_list = []
        for i_calc, var in enumerate(variables):
            deriv = sk_expr.diff(var)
            deriv=sympy.simplify(deriv); 
            gradient_list.append(deriv)
        jac_repl, jac_red = sympy.cse(gradient_list, optimizations='basic')
        jac_final_strs = [format_expr_str(expr) for expr in jac_red] if jac_red else [format_expr_str(g) for g in gradient_list]
        if len(jac_final_strs) != num_vars: raise ValueError("Jac expr count mismatch")
        jac_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in jac_repl]
        jac_cse_defs_str = textwrap.indent("\n".join(jac_cse_lines), "    ")
        jac_return_list_str = ",\n        ".join(jac_final_strs)
        jac_code = f"""
# --------------------------------------------------------------------------
# Jacobian Function ({func_base_name})
# Input: {wrapped_var_list}
# --------------------------------------------------------------------------
def {jac_func_name}(x_vec):
    \"\"\"Calculates Jacobian of S_{k} (n={n}).{common_docstring}
    Returns: list: Gradient vector.
    \"\"\"
    # Assign vars
{assign_vars_str}
    # Jacobian CSE Defs
{jac_cse_defs_str}
    # Final Gradient Calculation
    gradient = [
        {jac_return_list_str}
    ]
    return gradient
"""
        print(f"... Code for {jac_func_name} ready.")


    end_time_gen = time.time()
    print(f"... Generation for S_{k} group finished in {end_time_gen - start_time_gen:.2f} seconds.")
    return sk_code, sk_func_name, jac_code, jac_func_name

if __name__ == "__main__":
    N_TARGET = 3
    K_TARGETS = list(range(1, N_TARGET+1))
    output_filename = f"symbolic_minors_n{N_TARGET}.py"

    print(f"Starting code generation for N = {N_TARGET} (General Symm. Stochastic)")
    print(f"Target k values: {K_TARGETS}")
    print(f"Output file: {output_filename}")
    print("-" * 30)
    generated_blocks = []
    for k_target in K_TARGETS:
        print("-" * 30)
        sk_code, sk_name, jac_code, jac_name = generate_sk_jacobian_code(N_TARGET, k_target)

        if sk_code:
            generated_blocks.append({'k': k_target, 'sk_name': sk_name, 'sk_code': sk_code,
                                     'jac_name': jac_name, 'jac_code': jac_code})
        else: print(f"Skipping S_{k_target} due to S_k generation errors.")
    if generated_blocks:
        print("-" * 30); print(f"Writing generated functions to {output_filename}...")
        try:
            with open(output_filename, "w") as f:
                f.write("# -*- coding: utf-8 -*-\n")
                f.write(f"# Symbolic Functions for N = {N_TARGET} (General Symm. Stochastic)\n")
                f.write("# Generated by script\n\n")
                f.write("import math\n")
                f.write("import numpy as np\n\n")

                for block in generated_blocks:
                    f.write(block['sk_code'])
                    f.write("\n\n")
                    if block['jac_code']: 
                        f.write(block['jac_code'])
                        f.write("\n\n")
            print(f"Successfully wrote code to {output_filename}")

        except Exception as e: print(f"Error writing to file {output_filename}: {e}")
    else: 
        print("No functions generated.")

    print("-" * 30); print("Script finished.")