def save_spectra_results(results_x, results_y, filename="spectra_results.json"):
    string_results = []
    for point in zip(results_x, results_y):
        dict_val = {
            'x': str(point[0]),
            'y': str(point[1])
        }
        string_results.append(dict_val)
    with open(filename, 'w') as f:
            json.dump(string_results, f, indent=4)

if __name__ == '__main__':
    start_time = time.perf_counter()

    x, y = sp.symbols('x y', real=True)

    eq1 = x + y + 1
    eq2 = x*y

    min_x_points = []
    min_y_points = []
    for point in zip(x_values,min_y_values):
        eq1_temp = eq1 - point[0]
        eq2_temp = eq2 - point[1].fun

        # Solve the system
        solutions = sp.solve([eq1_temp, eq2_temp], [x, y])

        if solutions:
            for solution in solutions:
                min_x_points.append(solution[0])
                min_y_points.append(solution[1])

    max_x_points = []
    max_y_points = []
    for point in zip(x_values,max_y_values):
        eq1_temp = eq1 - point[0]
        eq2_temp = eq2 - point[1].fun

        # Solve the system
        solutions = sp.solve([eq1_temp, eq2_temp], [x, y])

        if solutions:
            for solution in solutions:
                max_x_points.append(solution[0])
                max_y_points.append(solution[1])

    # Record end time
    end_time_spec = time.perf_counter()

    # Calculate execution time
    time_taken_spec = end_time_spec - start_time

    print(f"Time taken to get coef data: {time_taken_spec:.6f} seconds")

    save_spectra_results(min_x_points, min_y_points, "min_points.json")
    save_spectra_results(max_x_points, max_y_points, "max_points.json")