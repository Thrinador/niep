import numpy as np
from scipy.optimize import LinearConstraint, minimize, NonlinearConstraint
import sympy as sp
from itertools import combinations
from math import comb
import tomli

with open("config.toml", "rb") as f:
    data = tomli.load(f)

# Global variables
n = data['global_data']['n']
tol = data['global_data']['tol']
initial_runs = data['global_data']['initial_runs']
subsequent_runs = data['global_data']['subsequent_runs']
type = data['global_data']['type']
num_variables = n**2 - n if type == 0 else comb(n, 2)

symbols = sp.symbols('a_:'+str(n**2))
matrix = sp.Matrix(n,n, symbols)
string_matrix = [['' for i in range(n)] for j in range(n)]

A = np.zeros((n, num_variables))

if type == 0:
    m=0
    for j in range(n):   
        for k in range(n):
            if k == j:
                matrix[j,k] = 'a_16'
                continue
            string_matrix[j][k] = '-a_'+str(m)
            matrix[j,k] = 'a_'+str(m)
            A[j][m-1] = 1
            m+=1

        row_sum = '1'
        for k in range(n):
            row_sum += string_matrix[j][k]
        matrix[j,j] = row_sum
else:
    m=0
    for j in range(n):   
        for k in range(j,n):
            if k == j:
                matrix[j,k] = 'a_16'
                continue
            string_matrix[j][k] = '-a_'+str(m)
            string_matrix[k][j] = '-a_'+str(m)
            matrix[j,k] = 'a_'+str(m)
            matrix[k,j] = 'a_'+str(m)
            A[j][m] = 1
            A[k][m] = 1
            m+=1

        row_sum = '1'
        for k in range(n):
            row_sum += string_matrix[j][k]
        matrix[j,j] = row_sum

matrix_constraints = LinearConstraint(A, np.zeros(n), np.ones(n))

def sum_matrix_minors(matrix, k):
    return sum(matrix[i, i].det() for i in combinations(range(n), k))

def run_function_with_const(loc, constraints = matrix_constraints):
    bounds = [(0.0, 1.0)] * num_variables

    count = 0
    best_result = None
    num_starts = initial_runs
    while count < 30:
        results = [minimize(funcs_of_principal_minors[loc], np.random.rand(num_variables), bounds=bounds, constraints=constraints, method='trust-constr', tol=tol, options={'maxiter': 300, 'initial_constr_penalty': 10000}) for _ in range(num_starts)]

        best_result = results[0]
        is_false = False
        for result in results:
            if result.success:
                is_false = True
                best_result = result
                break

        if is_false:
            for result in results:
                if result.fun <= best_result.fun and result.success:
                    best_result = result
            return best_result
        else:
            count += 1
            num_starts = subsequent_runs

    print(count)
    return best_result

def optimize_func(loc, eqs = []):
    if len(eqs) == 0:
        return run_function_with_const(loc)
    equals = []
    for i in range(0, len(eqs)):
        equals.append(NonlinearConstraint(
        lambda x: funcs_of_principal_minors[eqs[i][0]](x) - eqs[i][1],
        [0.0],
        [0.0]
    ))
    equals.append(matrix_constraints)
    return run_function_with_const(loc, equals)

funcs_of_principal_minors = tuple(
    sp.lambdify([symbols[0:num_variables]], sum_matrix_minors(matrix, k), 'numpy')
    for k in range(1, n+1)
) + tuple(
    sp.lambdify([symbols[0:num_variables]], -1*sum_matrix_minors(matrix, k), 'numpy')
    for k in range(1, n+1)
)