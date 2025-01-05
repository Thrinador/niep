def save_spectra_results(results_x, results_y, results_z, filename="spectra_results.json"):
    string_results = []
    for point in zip(results_x, results_y, results_z):
        dict_val = {
            'x': str(point[0]),
            'y': str(point[1]),
            'z': str(point[2])
        }
        string_results.append(dict_val)
    with open(filename, 'w') as f:
            json.dump(string_results, f, indent=4)