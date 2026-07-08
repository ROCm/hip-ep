#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import onnx
import os


def convert_to_external_data(input_onnx_path, output_onnx_path, size_threshold=1024):
    model = onnx.load_model(input_onnx_path)
    output_dir = os.path.dirname(output_onnx_path)
    output_data_file_name = os.path.basename(output_onnx_path).replace(".onnx", ".data")

    onnx.save_model(
        model,
        output_onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_data_file_name,
        size_threshold=size_threshold,
        convert_attribute=False,
    )
    print(
        f"onnx model converted to External Data format and saved as {output_onnx_path} and {output_dir}/{output_data_file_name}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python convert_onnx_to_external_data_mode.py <input_onnx> <output_onnx>"
        )
        sys.exit(1)

    input_onnx = sys.argv[1]
    if not os.path.exists(input_onnx):
        print(f"Input ONNX file {input_onnx} does not exist.")
        sys.exit(1)

    output_onnx = sys.argv[2]
    convert_to_external_data(input_onnx, output_onnx, 128)
