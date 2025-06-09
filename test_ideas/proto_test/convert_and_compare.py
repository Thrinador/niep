import json
import os
import data_block_pb2 # Import the generated code

def convert_json_to_proto(json_file, proto_file):
    """Reads a JSON file, converts it to protobuf, and saves it."""
    
    # 1. Read and parse the JSON file
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # 2. Create the top-level Protobuf message
    proto_array = data_block_pb2.DataBlockArray()

    # 3. Iterate through JSON objects and populate the Protobuf message
    for item in json_data:
        # Create a DataBlock message for each JSON object
        data_block = proto_array.items.add()
        
        # Populate the fields
        data_block.type = item['type']
        # The extend() method is used to add all elements from an iterable
        data_block.coefficients.extend(item['coefficients'])
        data_block.matrix.extend(item['matrix'])
        data_block.eigenvalues.extend(item['eigenvalues'])

    # 4. Serialize the Protobuf message to a binary string
    serialized_proto = proto_array.SerializeToString()

    # 5. Write the binary data to a file
    with open(proto_file, 'wb') as f:
        f.write(serialized_proto)
    print(f"Successfully converted {json_file} to {proto_file}.")


def compare_file_sizes(json_file, proto_file):
    """Compares and prints the sizes of the two files."""
    json_size = os.path.getsize(json_file)
    proto_size = os.path.getsize(proto_file)
    
    reduction = ((json_size - proto_size) / json_size) * 100

    print("\n--- File Size Comparison ---")
    print(f"JSON file size:      {json_size:,} bytes")
    print(f"Protobuf file size:  {proto_size:,} bytes")
    print(f"Size reduction:      {reduction:.2f}%")
    print("----------------------------\n")


if __name__ == "__main__":
    # Define file names
    json_input_file = "data.json"
    proto_output_file = "data.bin"

    # Step 2: Run the conversion
    convert_json_to_proto(json_input_file, proto_output_file)

    # Step 3: Compare the final file sizes
    compare_file_sizes(json_input_file, proto_output_file)