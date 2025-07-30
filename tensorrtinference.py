import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # This initializes CUDA context
import numpy as np
import time
import argparse
import wandb
from data_processing import preprocess_data
import gc
import yaml 
from main import get_dataset_info
import random
import torch

def seed_everything(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# --- Helper Function for TensorRT Engine Loading and Inference ---
class HostDeviceMem:
    """Helper class for managing host and device memory."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """
    Allocates host and device buffers for inputs and outputs based on
    the TensorRT engine's I/O tensors (for TensorRT 10.x API).
    """
    inputs = []  # HostDeviceMem objects for input tensors
    outputs = [] # HostDeviceMem objects for output tensors
    
    # This dictionary will store the device memory pointers, indexed by tensor name.
    device_memory_map = {} 
    
    stream = cuda.Stream() # CUDA stream for asynchronous operations

    # Iterate through all I/O tensors by their index (from 0 to num_io_tensors - 1)
    # engine.num_io_tensors gives the total count of input and output tensors in the engine.
    for i in range(engine.num_io_tensors):
        # Get the name of the tensor using its index.
        # In TensorRT 10.x, many tensor properties are queried using the tensor's name.
        tensor_name = engine.get_tensor_name(i)
        
        # Determine if the tensor is an input or an output.
        # Use get_tensor_mode and compare with trt.TensorIOMode enum.
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT 

        # Get the shape and data type of the tensor using its name.
        shape = engine.get_tensor_shape(tensor_name) 
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name)) 

        # Calculate the total number of elements for the buffer.
        # For fixed batch size engines, get_tensor_shape already includes the batch dimension
        # (e.g., (64, 1, 20) for your input).
        size = trt.volume(shape)

        # Allocate page-locked (pinned) memory on the host (CPU).
        # This improves performance for Host-to-Device (H2D) and Device-to-Host (D2H) transfers.
        host_mem = cuda.pagelocked_empty(size, dtype)
        
        # Allocate memory on the device (GPU).
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Store the device memory pointer in our map, using the tensor's name as the key.
        device_memory_map[tensor_name] = int(device_mem)

        # Create a HostDeviceMem object and append it to the appropriate list (inputs or outputs).
        if is_input:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    
    bindings = [0] * engine.num_io_tensors 

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        bindings[i] = device_memory_map[tensor_name] 

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    """
    Executes inference on the TensorRT engine.
    `context` is the TensorRT execution context.
    `bindings` are the device pointers for input/output.
    `inputs` is a list of HostDeviceMem objects for input data.
    `outputs` is a list of HostDeviceMem objects for output data.
    `stream` is a CUDA stream for async operations.
    """
    # Transfer input data to the GPU
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)

    # Execute inference
    # For fixed batch size, use execute_v2
    context.execute_v2(bindings=bindings)

    # Transfer output data from the GPU
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)

    # Synchronize the stream (important before accessing results on host)
    stream.synchronize()

    # Return the output data (NumPy arrays on host)
    return [out.host for out in outputs]


# --- Main Inference Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT Inference Benchmarking')
    parser.add_argument('--quantization', type=str, default="FP16", help="FP16 or INT8")
    parser.add_argument('--data', type=str, default='ISCXVPN2016', help='input dataset source (e.g., ISCXVPN2016 or MALAYAGT)')
    parser.add_argument('--path', type=str, default=None, help='Path to TensorRT engine. Overrides default path generation.')
    args = parser.parse_args()

    # --- Load Configuration from YAML ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Add dataset name from args to the config
    config['dataset_name'] = args.data
    data = args.data
    quant = args.quantization 

    # Determine engine path
    if args.path:
        user_path = args.path
        if '/' in user_path or '\\' in user_path:
            TRT_ENGINE_PATH = user_path
        else:
            TRT_ENGINE_PATH = f"saved_dict/{user_path}"
    else:
        # Default path if --path is not provided
        TRT_ENGINE_PATH = f"saved_dict/LiteNet_{data}_pruned_finetuned_embedding_{quant}.trt"

    print("TensorRT version:", trt.__version__)
    wandb.init(project="LiteNet-"+ data + "-inference", mode="disabled")
    seed_everything(134)
    
    # --- Configuration ---
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"
    INPUT_SHAPE = (config["batch_size"], config["features"])
    OUTPUT_SHAPE = (config["batch_size"], config["num_class"])
    NUM_INFERENCE_RUNS = 1000
    WARMUP_RUNS = 100
    if quant.lower() == "fp16":
        INPUT_DTYPE = np.float16
        OUTPUT_DTYPE = np.float16
    elif quant.lower() == "int8":
        INPUT_DTYPE = np.float32
        OUTPUT_DTYPE = np.float32
    else:
        INPUT_DTYPE = np.float32
        OUTPUT_DTYPE = np.float32

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    seed_everything(134)
    print(f"Loading TensorRT Engine from: {TRT_ENGINE_PATH}...")
    with open(TRT_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print("Engine loaded successfully.")
        
    
    context = engine.create_execution_context()

    # --- Dataset Loading and Preprocessing ---
    # Fetch classes and feature file from config.yaml
    classes, num_class, feature_file = get_dataset_info(config, args.data)
    # Load features
    try:
        most_important_list = np.load(feature_file)
    except FileNotFoundError:
        print(f"Error: Feature file '{feature_file}' not found.")
        exit()

    # Load raw data
    try:
        train_data_npy = np.load(f"dataset/{config['dataset_name']}/train.npy")
        test_data_npy = np.load(f"dataset/{config['dataset_name']}/test.npy")
        val_data_npy = np.load(f"dataset/{config['dataset_name']}/val.npy")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure data files are in '{config['dataset_name']}/' directory.")
        exit()
        
    # Preprocess data to get DataLoaders
    # IMPORTANT: Ensure prepare_dataloader is correctly imported and returns what's expected
    _, test_loader, _, pretime, avgpretime = preprocess_data(
        train_data_npy, test_data_npy, val_data_npy, most_important_list,
        config['batch_size'], config['dataset_name']
    )
    
    wandb.log({"preprocess_time": float(pretime)})
    wandb.log({"average_preprocess_time": float(avgpretime)})

    del train_data_npy, test_data_npy, val_data_npy, most_important_list, 
    gc.collect()

    # Allocate buffers for inputs and outputs once
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # Ensure the input buffer has the correct shape for numpy data (assuming single input)
    input_host_buffer = inputs[0].host.reshape(INPUT_SHAPE)

    # --- Warmup Phase ---
    print(f"\nStarting {WARMUP_RUNS} warmup runs...")

    try:
        # Get one batch for warmup. If test_loader is exhausted, re-initialize or use dummy data.
        warmup_batch_data, _ = next(iter(test_loader))
        if warmup_batch_data.is_cuda:
            warmup_batch_data = warmup_batch_data.cpu()
        warmup_input_np = warmup_batch_data.numpy().astype(INPUT_DTYPE).reshape(INPUT_SHAPE)
    except StopIteration:
        print("Warning: Test loader exhausted during warmup data retrieval. Using dummy data for warmup.")
        warmup_input_np = np.random.rand(*INPUT_SHAPE).astype(INPUT_DTYPE)

    print("input_host_buffer", input_host_buffer.dtype)
    print("warmup_input_np", warmup_input_np.dtype)
    np.copyto(input_host_buffer, warmup_input_np) # Copy dummy/first batch data to host buffer

    for _ in range(WARMUP_RUNS):
        # Just run inference, don't process results or time it
        do_inference(context, bindings, inputs, outputs, stream)
    print("Warmup complete.")
    
    # --- Benchmarking and Inference Loop ---
    # We will iterate through the test_loader
    total_batches = len(test_loader)
    
    print(f"\nStarting inference on {total_batches} batches from test_loader...")

    # Lists to store results and true labels if you want to calculate accuracy
    all_predictions = []
    all_true_labels = []

    start_time = time.time()
    num_processed_batches = 0

    for i, (batch_data, batch_labels) in enumerate(test_loader):
        
        # Move to CPU if it's on GPU 
        if batch_data.is_cuda:
            batch_data = batch_data.cpu()
        
        # Convert to NumPy and cast to FP16
        input_np_batch = batch_data.numpy().astype(INPUT_DTYPE)

        '''if 'debug_printed' not in locals():
            print("\n--- TensorRT DEBUG (First Batch) ---")
            print("Shape:", input_np_batch.shape)
            print("DType:", input_np_batch.dtype)
            print("Mean:", input_np_batch.mean())
            print("Std:", input_np_batch.std())
            print("First few values:", input_np_batch.flatten()[:5])
            print("-----------------------------------\n")
            debug_printed = True'''
        #input_np_batch = input_np_batch.reshape(INPUT_SHAPE)
        # Validate shape and data type before copying
        if input_np_batch.shape != INPUT_SHAPE:
            print(f"Warning: Batch {i} data shape mismatch! Expected {INPUT_SHAPE}, got {input_np_batch.shape}. Skipping batch.")
            continue # Skip this batch or handle error
        
        # 2. Copy NumPy array to the page-locked host buffer
        np.copyto(input_host_buffer, input_np_batch)

        # 3. Perform Inference
        trt_output_batch = do_inference(context, bindings, inputs, outputs, stream)
        
        # The result trt_output_batch is a list of NumPy arrays (one per output tensor)
        output_data_np = trt_output_batch[0].reshape(OUTPUT_SHAPE) # Reshape to expected output shape

        # 4. Post-processing
        # Assuming classification, apply argmax to get predicted labels
        predicted_labels = np.argmax(output_data_np, axis=1)
        
        # Store for overall metrics if needed
        all_predictions.extend(predicted_labels.tolist())
        all_true_labels.extend(batch_labels.numpy().tolist())

        num_processed_batches += 1

        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_batches} batches...")

    end_time = time.time()

    total_inference_time_s = (end_time - start_time)
    total_samples_processed = num_processed_batches * config["batch_size"]

    avg_inferencetime = (total_inference_time_s / total_samples_processed) if num_processed_batches > 0 else 0
    throughput_qps = total_samples_processed / total_inference_time_s if total_inference_time_s > 0 else 0
    throughput = 0.012/(float(avgpretime) + avg_inferencetime)
    print("\n--- Inference Summary ---")
    print(f"Total batches processed: {num_processed_batches}")
    print(f"Total samples processed: {total_samples_processed}")
    print(f"Total inference time: {total_inference_time_s:.4f} seconds")
    print(f"Average inference time per sample: {avg_inferencetime} s")
    print(f"Throughput: {throughput_qps:.2f} qps (queries per second)")
    print(f"Throughput (Mbps): {throughput:.2f} Mbps")
    # Optional: Calculate overall accuracy
    from sklearn.metrics import accuracy_score, classification_report

    print('--- Classification Report ---')
    print(classification_report(all_true_labels, all_predictions, target_names=classes, digits=4))
    if len(all_predictions) > 0:
        accuracy = accuracy_score(all_true_labels, all_predictions)
        print(f"Overall Accuracy on Test Set: {accuracy * 100:.2f}%")
        wandb.log({"test_accuracy_trt": accuracy})
        wandb.log({"trt_inferencetime_s": avg_inferencetime})
        wandb.log({"trt_throughput_qps": throughput_qps})
        wandb.log({"Throughput_mbps": throughput})
    else:
        print("No batches processed for accuracy calculation.")


    # Clean up
    del context
    del engine
    del inputs, outputs, bindings, stream
    gc.collect() # Ensure GPU memory is released

    wandb.finish()
    print("\nInference complete and resources released.")