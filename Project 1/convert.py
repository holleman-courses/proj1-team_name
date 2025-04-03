import h5py
import numpy as np
import sys


def convert_file_to_header(input_file, output_file, array_name="model_data"):
    with open(input_file, "rb") as f:
        data = f.read()

    # Format the byte data into a C array format.
    header_content = f"unsigned char {array_name}[] = {{\n"
    # Create comma separated hex values, 12 values per line for readability.
    line_length = 12
    hex_values = [f"0x{byte:02x}" for byte in data]
    for i in range(0, len(hex_values), line_length):
        line = ", ".join(hex_values[i:i + line_length])
        header_content += "  " + line + ",\n"
    header_content += "};\n"
    header_content += f"unsigned int {array_name}_len = {len(data)};\n"

    with open(output_file, "w") as f:
        f.write(header_content)
    print(f"Header file '{output_file}' created successfully.")


if __name__ == "__main__":
    # Usage: python convert_to_header.py input_filename output_filename
    if len(sys.argv) != 3:
        print("Usage: python convert_to_header.py <input_model.tflite> <output_model_data.h>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    convert_file_to_header(input_filename, output_filename)
