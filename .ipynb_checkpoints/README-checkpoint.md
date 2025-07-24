# ntc_inception

#Convert to .trt

# Bash script for linux

# Set dataset and quantization type

DATASET="ISCXVPN2016" # Specify dataset
QUANT="FP16" # Specify quantization
ONNX_MODEL_DIR="saved_dict"
ONNX_MODEL_PATH="${ONNX_MODEL_DIR}/LiteNet_${DATASET}_pruned_finetuned_${QUANT}.onnx"
TRT_ENGINE_PATH="${ONNX_MODEL_DIR}/LiteNet_${DATASET}_${QUANT}.trt"
INPUT_NAME="input"
BATCH_SIZE=64
SEQUENCE=1
FEATURES=20

trtexec --onnx=${ONNX_MODEL_PATH} \
        --saveEngine=${TRT_ENGINE_PATH} \
 --sparsity=enable \
 --fp16 \
 --useCudaGraph\
 --shapes=${INPUT_NAME}:${BATCH_SIZE}x${SEQUENCE}x${FEATURES}

# For PowerShell use this

$DATASET = "ISCXVPN2016"           # Specify dataset
$QUANT = "FP16" # Specify quantization (FP16 or INT8)
$ONNX_MODEL_DIR = "saved_dict"
$ONNX*MODEL_PATH = "$ONNX_MODEL_DIR/LiteNet*${DATASET}_pruned_finetuned_${QUANT}.onnx"
$TRT_ENGINE_PATH = "$ONNX*MODEL_DIR/LiteNet*${DATASET}_${QUANT}.trt"
$INPUT_NAME = "input"
$BATCH_SIZE = 64
$SEQUENCE = 1
$FEATURES = 20

trtexec --onnx=$ONNX_MODEL_PATH `
        --saveEngine=$TRT_ENGINE_PATH `        --sparsity=enable`
--fp16 `
--shapes="${INPUT_NAME}:${BATCH_SIZE}x${SEQUENCE}x${FEATURES}"
