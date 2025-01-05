import json
from sympy import symbols, nonlinsolve, Reals
from pathos.pools import ProcessPool as Pool

# Define the variables
x, y, z = symbols('x y z', real=True)

# Define the equations
eq1 = x + y + z + 1
eq2 = x + y + z + x*y + x*z + y*z
eq3 = x*y + x*z + y*z + x*y*z

errors = []

def solve_equation(point):
    try:
        solutions = nonlinsolve([eq1 - point[0], eq2 - point[1], eq3 - point[2]], [x, y, z]) & Reals ** 3
        if solutions:
            return [[str(x) for x in sol] for sol in solutions]
    except:
        errors.append(point)
    
    return None

def save_results(results, filename):
    with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

with open('data/ds-sniep/ds-sniep_max_values_4_2_2.json') as f:
    max_vals = json.load(f)

with open('data/ds-sniep/ds-sniep_min_values_4_2_2.json') as f:
    min_vals = json.load(f)

X = [point[0] for point in max_vals]
Y = [point[1] for point in max_vals]
Z_max = [point[2]['output'] for point in max_vals]
Z_min = [point[2]['output'] for point in min_vals]

with Pool() as pool:
    min_points = pool.map(lambda x: solve_equation(x), zip(X,Y,Z_min))
    max_points = pool.map(lambda x: solve_equation(x), zip(X,Y,Z_max))

save_results(min_points, "data/ds-sniep/test_min.json")
save_results(max_points, "data/ds-sniep/test_max.json")