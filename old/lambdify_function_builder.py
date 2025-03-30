import sympy as sp
from itertools import combinations
from math import comb
import tomli
import dill
import sys

sys.setrecursionlimit(10000)

def sum_matrix_minors(matrix, k):
    return sum(matrix[i, i].det() for i in combinations(range(n), k))

if __name__ == '__main__':
    # Build config variables
    with open("config.toml", "rb") as f:
        data = tomli.load(f)
    n = data['global_data']['n']
    type = data['global_data']['type']
    num_variables = n**2 - n if type == 0 else comb(n, 2) if type == 1 else comb(n+1, 2)

    # Setup symbolic matrix to be working with.
    symbols = sp.symbols('a_:'+str(n**2))
    matrix = sp.Matrix(n,n, symbols)
    string_matrix = [['' for i in range(n)] for j in range(n)]
    m=0
    if type == 0:
        for j in range(n):   
            for k in range(n):
                if k != j:
                    string_matrix[j][k] = '-a_'+str(m)
                    matrix[j,k] = 'a_'+str(m)
                    m+=1
            row_sum = '1'
            for k in range(n):
                row_sum += string_matrix[j][k]
            matrix[j,j] = row_sum
    elif type == 1:
        for j in range(n):   
            for k in range(j,n):
                if k != j:
                    string_matrix[j][k] = '-a_'+str(m)
                    string_matrix[k][j] = '-a_'+str(m)
                    matrix[j,k] = 'a_'+str(m)
                    matrix[k,j] = 'a_'+str(m)
                    m+=1

            row_sum = '1'
            for k in range(n):
                row_sum += string_matrix[j][k]
            matrix[j,j] = row_sum
    elif type == 2:
        for j in range(n):   
            for k in range(j,n):
                matrix[j,k] = 'a_'+str(m)
                matrix[k,j] = 'a_'+str(m)
                m+=1

    print("Building functions")
    funcs_of_principal_minors = tuple(
        sp.lambdify([symbols[0:num_variables]], sum_matrix_minors(matrix, k+1), 'numpy')
        for k in range(n)
    ) + tuple(
        sp.lambdify([symbols[0:num_variables]], -1*sum_matrix_minors(matrix, k+1), 'numpy')
        for k in range(n)
    )
    print("Functions have been built")
    dill.settings['recurse'] = True
    if type == 0:
        dill.dump(list(funcs_of_principal_minors), open("lambdified_functions/niep_" + str(n), "wb"))
    elif type == 1:
        dill.dump(list(funcs_of_principal_minors), open("lambdified_functions/ds-sniep_" + str(n), "wb"))
    elif type == 2:
        dill.dump(list(funcs_of_principal_minors), open("lambdified_functions/sniep_" + str(n), "wb"))

