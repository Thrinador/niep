import json
from sympy import symbols, nonlinsolve, Reals
from pathos.pools import ProcessPool as Pool
import tomli
import numpy as np
import time


# Build config variables
with open("eig_config.toml", "rb") as f:
    data = tomli.load(f)

type = data['global_data']['type']
n = data['global_data']['n']

errors = []

def build_matrix(array):
    matrix = np.zeros((n,n))
    m = 0
    if type == 0:
        for j in range(n):
            for k in range(n):
                if k == j:
                    continue
                matrix[j][k] = array[m]
                m+=1
            
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1 - row_sums[j]
    else:
        for j in range(n):
            for k in range(j, n):
                if k == j:
                    continue
                matrix[j][k] = array[m]
                matrix[k][j] = array[m]
                m+=1
            
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1 - row_sums[j]
    return matrix

def find_eigenvalues(matrix):
    eigvals = list(np.linalg.eigvals(matrix))

    # Find the closest eigenvalue to 1 and remove it.
    spot = 0
    for i in range(len(eigvals)):
        if abs(eigvals[i] - 1) < abs(eigvals[spot] - 1):
            spot = i
    del eigvals[spot]
    eigvals.sort(reverse=True)
    return eigvals


def save_results(results, filename):
    with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

with open('data/ds-sniep/ds-sniep_max_values_4_4_4.json') as f:
    max_vals = json.load(f)

with open('data/ds-sniep/ds-sniep_min_values_4_4_4.json') as f:
    min_vals = json.load(f)

start_time = time.perf_counter()

max_matrices = [build_matrix(x[2]['matrix']) for x in max_vals]
min_matrices = [build_matrix(x[2]['matrix']) for x in min_vals]

print(f"Matrices built in {time.perf_counter() - start_time:.6f} seconds")
start_time = time.perf_counter()

max_eigvals = [find_eigenvalues(x) for x in max_matrices]
min_eigvals = [find_eigenvalues(x) for x in min_matrices]

print(f"Eigenvalues found in {time.perf_counter() - start_time:.6f} seconds")

save_results(max_eigvals, "data/ds-sniep/test2_min.json")
save_results(min_eigvals, "data/ds-sniep/test2_max.json")
