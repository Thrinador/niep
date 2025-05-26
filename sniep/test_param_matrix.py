import sympy

# Define symbolic variables
# Using 'lamda' because 'lambda' is a reserved keyword in Python
a, b, c, lamda = sympy.symbols('a b c lambda', real=True) 

# Define the matrix M with symbolic entries
M = sympy.Matrix([
    [1-a-c, 0,     a,     0,     c],
    [0,     1-a-b, 0,     b,     a],
    [a,     0,     1-a-b, b,     0],
    [0,     b,     b,     1-2*b, 0],
    [c,     a,     0,     0,     1-a-c]
])

# Define the 5x5 identity matrix
I = sympy.eye(5)

# Form the matrix M - lambda*I
M_minus_lambda_I = M - lamda * I

print("Calculating the characteristic polynomial det(M - lambda*I)...")
print("This might take a moment.")

# Calculate the determinant symbolically
try:
    char_poly_expr = M_minus_lambda_I.det()
    print("Determinant calculation complete. Expanding the polynomial...")
    
    # Expand the polynomial for a standard form
    # Use simplify or expand. Expand might be more direct for polynomial structure.
    expanded_char_poly = sympy.expand(char_poly_expr)
    
    print("\nThe characteristic polynomial P(a, b, c, lambda) is:")
    # Use pretty printing for better readability if available
    try:
        sympy.init_printing(use_unicode=True) # Use unicode characters for prettier output
        sympy.pprint(expanded_char_poly)
    except Exception: # Fallback if pretty printing fails
        print(expanded_char_poly)
    print("\nThis polynomial equation P(a, b, c, lambda) = 0 defines the surface containing all eigenvalues.")

    # --- Factor out the known (lambda - 1) factor ---
    print("\nFactoring the characteristic polynomial...")
    
    # Convert the expression to a Polynomial object in variable 'lamda' for robust factoring/division
    poly_in_lambda = sympy.Poly(expanded_char_poly, lamda)
    
    # Factor the polynomial
    factored_poly = sympy.factor(poly_in_lambda)
    
    print("\nFactored characteristic polynomial:")
    try:
        sympy.pprint(factored_poly)
    except Exception:
        print(factored_poly)
        
    # --- Isolate the polynomial for the other 4 eigenvalues ---
    # We know (lambda - 1) is a factor. Divide it out.
    # Using polynomial division (quo function)
    remaining_poly_factor = sympy.quo(poly_in_lambda, (lamda - 1))
    
    print("\nThe polynomial factor Q(a, b, c, lambda) whose roots are the other 4 eigenvalues is:")
    try:
        # Expand the result for potentially easier reading
        sympy.pprint(sympy.expand(remaining_poly_factor.as_expr()))
    except Exception:
        print(sympy.expand(remaining_poly_factor.as_expr()))
    print("\nThe equation Q(a, b, c, lambda) = 0 defines the surface for the eigenvalues other than lambda=1.")

except Exception as e:
    print(f"\nAn error occurred during symbolic calculation: {e}")
    print("Please ensure SymPy is installed correctly ('pip install sympy').")