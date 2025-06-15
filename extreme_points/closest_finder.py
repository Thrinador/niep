import json
import numpy as np

def find_closest_point(json_file, target_point):
    """
    Finds the data point in a JSON file that is closest to a target point.

    Args:
        json_file (str): The path to the JSON file.
        target_point (list): The target point to compare against.

    Returns:
        dict: The block of data corresponding to the closest point, or None if not found.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file}' is not a valid JSON file.")
        return None

    target_point = np.array(target_point)
    closest_point_data = None
    min_distance = float('inf')

    for block in data:
        if "eigenvalues" in block and len(block["eigenvalues"]) == 5:
            eigenvalues = np.array(block["eigenvalues"])
            distance = np.linalg.norm(eigenvalues - target_point)

            if distance < min_distance:
                min_distance = distance
                closest_point_data = block

    return closest_point_data, min_distance

if __name__ == '__main__':
    # Define the target point
    target = [0.25, 0.25, 0.2, -0.75, -0.75]

    # Specify the JSON file name
    json_filename = '../sub_sniep/data/sub_sniep_n6_dims37_37_37_37.json'

    # Find the closest point
    closest_block, distance = find_closest_point(json_filename, target)

    # Print the results
    if closest_block:
        print("Target Point:")
        print(target)
        print("\nClosest Data Point Found:")
        print(json.dumps(closest_block, indent=4))
        print(f"\nEuclidean Distance: {distance}")