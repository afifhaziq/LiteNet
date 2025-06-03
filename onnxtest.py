import onnxruntime
import numpy as np

# Define input shape based on model.py's dummy_input
# Example: (batch_size, sequence_length, features)
# Replace with your actual model's input shape!
BATCH_SIZE = 64
SEQUENCE_LENGTH = 1
FEATURES = 20 

# Dummy input matching the expected shape and type (e.g., float32)
# Ensure this matches the dummy_input you used for ONNX export
dummy_input_onnx = np.random.rand(BATCH_SIZE, SEQUENCE_LENGTH, FEATURES).astype(np.float32)

# Path to your ONNX model file
onnx_model_path = 'saved_dict/NtCNN_ISCXVPN2016_20Features_best_model_pruned_finetuned.onnx' # Make sure this is the correct path to your exported ONNX model

try:
    # Load the ONNX model
    sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    print(f"ONNX model loaded successfully from: {onnx_model_path}")

    # Get input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX Input Name: {input_name}")
    print(f"ONNX Output Name: {output_name}")

    # Run inference
    output = sess.run([output_name], {input_name: dummy_input_onnx})
    print("ONNX Runtime inference successful!")
    print(f"Output shape: {output[0].shape}")

except Exception as e:
    print(f"Error loading or running ONNX model: {e}")