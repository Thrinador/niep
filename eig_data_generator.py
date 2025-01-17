import json
import tomli
import numpy as np
import time

# Build config variables
with open("config.toml", "rb") as f:
    data = tomli.load(f)

type = data['global_data']['type']
n = data['global_data']['n']

def build_matrix(array):
    matrix = np.zeros((n,n))
    m = 0
    if type == 0:
        for j in range(n):
            for k in range(n):
                if k != j:
                    matrix[j][k] = array[m]
                    m+=1
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1 - row_sums[j]
    else:
        for j in range(n):
            for k in range(j, n):
                if k != j:
                    matrix[j][k] = array[m]
                    matrix[k][j] = array[m]
                    m+=1
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1 - row_sums[j]
    return matrix

def find_eigenvalues(matrix):
    eigvals = list(np.linalg.eigvals(matrix))
    eigvals.sort(reverse=True)
    del eigvals[0]
    return eigvals

def save_results(results, filename):
    with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

start_time = time.perf_counter()
with open('data/ds-sniep/ds-sniep_max_values_4_100_100.json') as f:
    max_matrices = [build_matrix(x[2]['matrix']) for x in json.load(f)]
with open('data/ds-sniep/ds-sniep_min_values_4_100_100.json') as f:
    min_matrices = [build_matrix(x[2]['matrix']) for x in json.load(f)]
print(f"Matrices loaded and built in {time.perf_counter() - start_time:.6f} seconds")

start_time = time.perf_counter()
max_eigvals = [[find_eigenvalues(x), x.tolist()] for x in max_matrices]
min_eigvals = [[find_eigenvalues(x), x.tolist()] for x in min_matrices]
print(f"Eigenvalues found in {time.perf_counter() - start_time:.6f} seconds")

save_results(max_eigvals, "data/ds-sniep/ds-sniep_max_eig_4_100_100.json")
save_results(min_eigvals, "data/ds-sniep/ds-sniep_min_eig_4_100_100.json")
print("Data saved")
