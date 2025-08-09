import onnxruntime as ort
import numpy as np
from main import get_dataset_info
import gc
import yaml 
from data_processing import preprocess_data

# Define input shape based on model.py's dummy_input
# Example: (batch_size, sequence_length, features)
# Replace with your actual model's input shape!

data = "MALAYAGT"

# --- Load Configuration from YAML ---
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
INPUT_SHAPE = (config["batch_size"], config["features"])
classes, num_class, feature_file = get_dataset_info(config, data)
    # Load features
try:
    most_important_list = np.load(feature_file)
except FileNotFoundError:
    print(f"Error: Feature file '{feature_file}' not found.")
    exit()

    # Load raw data
try:
    train_data_npy = np.load(f"dataset/{data}/train.npy")
    test_data_npy = np.load(f"dataset/{data}/test.npy")
    val_data_npy = np.load(f"dataset/{data}/val.npy")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please ensure data files are in '{data}/' directory.")
    exit()
        
    # Preprocess data to get DataLoaders
    # IMPORTANT: Ensure prepare_dataloader is correctly imported and returns what's expected
train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(train_data_npy, test_data_npy, val_data_npy, most_important_list,config['batch_size'], data)
    
del train_data_npy, test_data_npy, val_data_npy, most_important_list, train_loader, val_loader
gc.collect()
# Dummy input matching the expected shape and type (e.g., float32)
# Ensure this matches the dummy_input you used for ONNX export
#dummy_input_onnx = np.random.rand(BATCH_SIZE, SEQUENCE_LENGTH, FEATURES).astype(np.float32)
#dummy_input_onnx = np.load('dataset/ISCXVPN2016/test.npy')



# Path to your ONNX model file
onnx_model_path = f'saved_dict/LiteNet_{data}_FullPruned_finetuned_embedding_FP16.onnx' # Make sure this is the correct path to your exported ONNX model

# Load your ONNX model
onnx_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Get model input and output names
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Assume 'val_loader' is your PyTorch validation data loader
correct = 0
total = 0

for images, labels in test_loader:
    # IMPORTANT: Ensure your preprocessing is identical to PyTorch's
    # The input numpy array must be in the correct format (e.g., NCHW) and type (e.g., np.float32)
    input_data = images.numpy().astype(np.float16)

    # Run inference
    result = onnx_session.run([output_name], {input_name: input_data})[0]

    # Calculate accuracy
    predicted = np.argmax(result, axis=1)
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum().item()

print(f"ONNX Model Accuracy: {100 * correct / total:.2f}%")

'''for i, (batch_data, batch_labels) in enumerate(test_loader):
       
        if batch_data.is_cuda:
            batch_data = batch_data.cpu()
        
        # Convert to NumPy and cast to FP16
        dummy_input_onnx = batch_data.numpy().astype(np.float16)

        if 'debug_printed' not in locals():
            print("\n--- TensorRT DEBUG (First Batch) ---")
            print("Shape:", input_np_batch.shape)
            print("DType:", input_np_batch.dtype)
            print("Mean:", input_np_batch.mean())
            print("Std:", input_np_batch.std())
            print("First few values:", input_np_batch.flatten()[:5])
            print("-----------------------------------\n")
            debug_printed = True
       
        if dummy_input_onnx.shape != INPUT_SHAPE:
            print(f"Warning: Batch {i} data shape mismatch! Expected {INPUT_SHAPE}, got {dummy_input_onnx.shape}. Skipping batch.")
            continue # Skip this batch or handle error
        
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
    print(f"Error loading or running ONNX model: {e}")'''