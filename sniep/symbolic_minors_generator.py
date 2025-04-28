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
def generate_sk_jacobian_hessian_code(n, k):
    """
    Generates Python code strings for S_k, its Jacobian, and its Hessian
    for a general symmetric stochastic matrix (parameterized by upper triangle).
    Includes an option for potentially slow aggressive simplification.
    """
    # --- CONFIGURATION FOR THIS FUNCTION ---
    AGGRESSIVE_SIMPLIFY = False
    # ---

    if k < 1 or k > n: print(f"Error: k={k} out of range for n={n}."); return None, None, None, None, None, None

    start_time_gen = time.time()
    # Define function names consistently
    func_base_name = f"S{k}_n{n}"; sk_func_name = f"calculate_{func_base_name}"
    jac_func_name = f"calculate_{func_base_name}_jacobian"; hess_func_name = f"calculate_{func_base_name}_hessian"

    # 1. Define ALL symbols for the strict upper triangle
    variables = []; variable_map = {}; var_names = []; idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            var_name = f'x_{i}_{j}'; sym = sympy.symbols(var_name)
            variables.append(sym); var_names.append(var_name); variable_map[(i, j)] = sym; idx += 1
    num_vars = len(variables)
    print(f"Generating S_{k}, Jac, Hess for N={n} general symm. stochastic ({num_vars} variables)...")
    if AGGRESSIVE_SIMPLIFY: print("!!! AGGRESSIVE SIMPLIFICATION ENABLED (MAY BE VERY SLOW) !!!")

    # Initialize return values
    sk_code, jac_code, hess_code = None, None, None
    sk_expr = None

    # 2. Build symbolic matrix & Calculate S_k (within a try block)
    try:
        print("... Constructing symbolic matrix...")
        M = sympy.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n): val = variable_map[(i, j)]; M[i, j] = val; M[j, i] = val
        print("... Calculating symbolic diagonal elements...")
        for i in range(n):
            off_diag_sum = sympy.sympify(0); [off_diag_sum := off_diag_sum + M[i,c] for c in range(n) if i!=c]; # Requires Python 3.8+
            diag_val = 1 - off_diag_sum
            if AGGRESSIVE_SIMPLIFY: print(f"      Simp M[{i},{i}]..."); simp_start=time.time(); diag_val=sympy.simplify(diag_val); print(f"->took {time.time()-simp_start:.2f}s")
            M[i, i] = diag_val
        print("... Symbolic matrix constructed.")

        print("... Calculating symbolic S_k expression...")
        sk_expr = sympy.sympify(0); indices = range(n); total_minors = sympy.binomial(n, k)
        for i_calc, subset_indices in enumerate(itertools.combinations(indices, k)):
            submatrix = M[list(subset_indices), list(subset_indices)]; det_expr = sympy.expand(submatrix.det()); sk_expr += det_expr
            if (i_calc+1)%max(1,total_minors//10)==0 or (i_calc+1)==total_minors: print(f"    S_k Det {i_calc+1}/{total_minors}...")
        if AGGRESSIVE_SIMPLIFY: print("... Aggressively simplifying S_k (SLOW)..."); simp_start=time.time(); sk_expr=sympy.simplify(sk_expr); print(f"->took {time.time()-simp_start:.2f}s")
        print("... Symbolic S_k calculation complete.")

    except Exception as e: print(f"ERROR S_k calc: {e}"); traceback.print_exc(); return None, None, None, None, None, None

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
        print("... Applying CSE to S_k...")
        sk_repl, sk_red = sympy.cse(sk_expr, optimizations='basic')
        sk_final_str = format_expr_str(sk_red[0]) if sk_red else format_expr_str(sk_expr)
        sk_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in sk_repl]
        sk_cse_defs_str = textwrap.indent("\n".join(sk_cse_lines), "    ")
        print("... Generating S_k function code...")
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
        try: # Try Jacobian code generation
            print("... Calculating symbolic Jacobian...")
            gradient_list = []
            for i_calc, var in enumerate(variables):
                deriv = sk_expr.diff(var)
                if AGGRESSIVE_SIMPLIFY: print(f"    Simp dS{k}/d{var}..."); simp_start=time.time(); deriv=sympy.simplify(deriv); print(f" -> {time.time()-simp_start:.2f}s")
                gradient_list.append(deriv)
            print("... Applying CSE to Jacobian...")
            jac_repl, jac_red = sympy.cse(gradient_list, optimizations='basic')
            jac_final_strs = [format_expr_str(expr) for expr in jac_red] if jac_red else [format_expr_str(g) for g in gradient_list]
            if len(jac_final_strs) != num_vars: raise ValueError("Jac expr count mismatch")
            jac_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in jac_repl]
            jac_cse_defs_str = textwrap.indent("\n".join(jac_cse_lines), "    ")
            jac_return_list_str = ",\n        ".join(jac_final_strs)
            print("... Generating Jacobian function code...")
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
        except Exception as e: print(f"Error Jacobian CodeGen: {e}"); jac_code = None

    # --- Generate Hessian Function Code ---
    if sk_code and sk_expr is not None:
        try: # Try Hessian code generation
            print("... Calculating symbolic Hessian (SLOW)...")
            hess_start = time.time(); hessian_sym = sympy.hessian(sk_expr, variables); print(f"... Hess calc took {time.time()-hess_start:.2f}s.")
            if AGGRESSIVE_SIMPLIFY:
                print("... Aggressively simplifying Hessian (EXTREMELY SLOW)...")
                simp_start=time.time(); rows,cols=hessian_sym.shape; simplified_hess=sympy.zeros(rows,cols); total_elems=rows*cols; elem_count=0
                for r in range(rows):
                    for c in range(cols): elem_count+=1;
                    if hessian_sym[r,c]!=0: simplified_hess[r,c]=sympy.simplify(hessian_sym[r,c])
                    if elem_count%max(1,total_elems//10)==0 or elem_count==total_elems: print(f"    Simp Hess elem {elem_count}/{total_elems}...")
                hessian_sym = simplified_hess; print(f"    Hess simp took {time.time()-simp_start:.2f}s")
            print("... Applying CSE to Hessian..."); hess_cse_start = time.time()
            hess_repl, hess_red = sympy.cse(hessian_sym, optimizations='basic')
            print(f"... Hess CSE took {time.time() - hess_cse_start:.2f}s.")
            hess_cse_lines = [f"{sym} = {format_expr_str(expr)}" for sym, expr in hess_repl]
            hess_cse_defs_str = textwrap.indent("\n".join(hess_cse_lines), "    ")
            hess_final_strs = [format_expr_str(expr) for expr in hess_red] if hess_red else []
            expected_hess_elems = num_vars * num_vars
            if len(hess_final_strs) != expected_hess_elems:
                 print(f"    W: CSE Hess expr count mismatch. Using direct.");
                 hess_final_strs=[format_expr_str(hessian_sym[r, c]) for r in range(num_vars) for c in range(num_vars)]
                 if len(hess_final_strs) != expected_hess_elems: raise ValueError("Hess expr count mismatch")
            hess_calc_lines = []; idx = 0
            for r in range(num_vars):
                for c in range(num_vars): hess_calc_lines.append(f"    hessian_np[{r}, {c}] = {hess_final_strs[idx]}"); idx += 1
            hess_calc_str = "\n".join(hess_calc_lines)
            print("... Generating Hessian function code...")
            hess_code = f"""
# --------------------------------------------------------------------------
# Hessian Function ({func_base_name})
# Input: {wrapped_var_list}
# --------------------------------------------------------------------------
def {hess_func_name}(x_vec):
    \"\"\"Calculates Hessian matrix of S_{k} (n={n}).{common_docstring}
    Returns: numpy.ndarray: Hessian matrix ({num_vars}x{num_vars}).
    \"\"\"
    # Assign vars
{assign_vars_str}
    # Hessian CSE Defs
{hess_cse_defs_str}
    # Final Hessian Calculation
    hessian_np = np.zeros(({num_vars}, {num_vars})) # Initialize
{hess_calc_str}
    return hessian_np
"""
            print(f"... Code for {hess_func_name} ready.")
        except Exception as e: # <<< Correctly indented except block for Hessian try
            print(f"Error Hessian CodeGen: {e}")
            print("!!! Hessian generation FAILED. !!!");
            hess_code = None # <<< FIX: Only nullify code, not name

    # --- Finished ---
    end_time_gen = time.time()
    print(f"... Generation for S_{k} group finished in {end_time_gen - start_time_gen:.2f} seconds.")
    # --- FIX: Use consistent names in return ---
    return sk_code, sk_func_name, jac_code, jac_func_name, hess_code, hess_func_name
    # --- End Correction ---

# --- Main Script Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    N_TARGET = 6
    K_TARGETS = list(range(1, N_TARGET))
    output_filename = f"symbolic_minors_n{N_TARGET}.py"
    # --- End Configuration ---

    print(f"Starting code generation for N = {N_TARGET} (General Symm. Stochastic)")
    # ... (rest of main block: loop k_targets, call generator, write file, add tests) ...
    # Ensure the test code uses the updated general num_vars and 1/N test case
    print(f"Target k values: {K_TARGETS}"); print(f"Output file: {output_filename}")
    if N_TARGET >= 4: print("\nWARNING: Hessian generation for N>=4 can be VERY slow and memory intensive!\n")
    print("-" * 30)
    generated_blocks = []
    for k_target in K_TARGETS:
        print("-" * 30)
        sk_code, sk_name, jac_code, jac_name, hess_code, hess_name = \
            generate_sk_jacobian_hessian_code(N_TARGET, k_target)
        if sk_code:
            generated_blocks.append({'k': k_target, 'sk_name': sk_name, 'sk_code': sk_code,
                                     'jac_name': jac_name, 'jac_code': jac_code,
                                     'hess_name': hess_name, 'hess_code': hess_code})
        else: print(f"Skipping S_{k_target} due to S_k generation errors.")
    if generated_blocks:
        print("-" * 30); print(f"Writing generated functions to {output_filename}...")
        try:
            with open(output_filename, "w") as f:
                f.write("# -*- coding: utf-8 -*-\n"); f.write(f"# Symbolic Functions for N = {N_TARGET} (General Symm. Stochastic)\n")
                f.write("# Generated by script\n\n"); f.write("import math\n"); f.write("import numpy as np\n\n")
                for block in generated_blocks:
                    f.write(block['sk_code']); f.write("\n\n");
                    if block['jac_code']: f.write(block['jac_code']); f.write("\n\n")
                    if block['hess_code']: f.write(block['hess_code']); f.write("\n\n")
            print(f"Successfully wrote code to {output_filename}")
            # --- Add Test Code (using 1/N matrix case) ---
            print("Adding basic test usage example...")
            num_vars_test = N_TARGET * (N_TARGET - 1) // 2 # General num_vars
            x_test_val = 1.0 / N_TARGET; x_test_vec = [x_test_val] * num_vars_test
            x_test_tuple_str = str(tuple(x_test_vec))
            s_k_expected = {1: 1.0}; [s_k_expected.update({k_exp: 0.0}) for k_exp in range(2, N_TARGET + 1)] # S1=1, Sk=0 for k>1
            test_code = f"""
# --- Basic Test Usage ---
if __name__ == '__main__':
    print("\\n" + "-"*20); print("--- Running Basic Tests for N={N_TARGET} (General) ---"); print("-" * 20)
    x_test = {x_test_tuple_str} # Test vector for the 1/N matrix
    print(f"Test input vector (len {num_vars_test}): {{x_test}}")\n
    s_k_expected = {s_k_expected}
"""
            for block in generated_blocks:
                 k_test=block['k']; sk_name=block['sk_name']; jac_name=block['jac_name']; hess_name=block['hess_name']
                 test_code += f"""
    print("\\n--- Testing S_{k_test} ---")
    # Value
    val = None
    try:
        val = {sk_name}(x_test)
        exp_val = s_k_expected.get({k_test}, None)
        print(f"  {sk_name}(x_test) = {{val:.6e}}")
        if exp_val is not None:
            check_result = 'PASS' if abs(val - exp_val) < 1e-9 else 'FAIL'
            print(f"    Check vs Expected ({{exp_val:.1f}}): {{check_result}}")
        else: print("    (No expected value for check)")
    except Exception as e: print(f"  Error testing {sk_name}: {{e}}")
"""
                 if jac_name: test_code += f"""
    # Jacobian
    try: jac = {jac_name}(x_test); print(f"  {jac_name}(x_test) returned vector length {{len(jac)}}"); is_zero = all(abs(j) < 1e-9 for j in jac); print(f"    Check if zero vector: {{is_zero}}")
    except Exception as e: print(f"  Error testing {jac_name}: {{e}}")
"""
                 if hess_name: test_code += f"""
    # Hessian
    try: hess = {hess_name}(x_test); print(f"  {hess_name}(x_test) returned matrix shape {{hess.shape if hasattr(hess,'shape') else 'N/A'}}")
    except Exception as e: print(f"  Error testing {hess_name}: {{e}}")
"""
            with open(output_filename, "a") as f: f.write("\n" + test_code)
            print("Test code added.")
        except Exception as e: print(f"Error writing to file {output_filename}: {e}")
    else: print("No functions generated.")
    print("-" * 30); print("Script finished.")