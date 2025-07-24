import torch
import copy
from model import QuantizedLiteNet

# --- FP16 Quantization Function ---
def quantize_fp16(model):
    print("\n--- Applying FP16 Quantization ---")
    model.half()
    print("Model converted to FP16.")
    return model

# --- INT8 Static Quantization Function ---
def quantize_int8_static(model, train_loader, device):
    print("\n--- Applying INT8 Static Quantization ---")
    
    # INT8 quantization must be done on the CPU
    quant_model = copy.deepcopy(model).to('cpu')
    quant_model.eval()

    print("Fusing model modules...")
    layers_to_fuse = [
        ['branch1x1.0', 'branch1x1.1'],
        ['branch3x3.1', 'branch3x3.2'],
        ['branch5x5.1', 'branch5x5.2'],
        ['branch_pool.1', 'branch_pool.2'],
        ['fc2', 'activation6']
    ]
    try:
        torch.ao.quantization.fuse_modules(quant_model, layers_to_fuse, inplace=True)
        print("Fusion complete.")
    except Exception as e:
        print(f"Could not fuse modules: {e}")

    quant_model = QuantizedLiteNet(quant_model)
    quant_model.eval()
    
    quant_model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    print(f"Using quantization config: {quant_model.qconfig}")

    print("Preparing for calibration...")
    torch.ao.quantization.prepare(quant_model, inplace=True)
    
    print("Calibrating model with training data (10 batches)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            images = images.to('cpu').float()
            quant_model(images)
            if i >= 10:
                break
    print("Calibration complete.")

    torch.ao.quantization.convert(quant_model, inplace=True)
    print("Model converted to INT8.")
    
    return quant_model.to(device) 