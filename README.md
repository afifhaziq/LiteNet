# LiteNet

This project provides a complete, end-to-end pipeline for training, optimizing, and deploying a neural network for LiteNet, a Network Traffic Classification (NTC) model. The pipeline includes feature selection with SHAP, semi-structured sparse pruning, quantization (FP16/INT8), and conversion to a TensorRT engine for high-performance inference.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Pipeline](#project-pipeline)
- [How to Use](#how-to-use)
  - [1. `config.yaml`](#1-configyaml)
  - [2. `feature_selection.py`](#2-feature_selectionpy)
  - [3. `main.py`](#3-mainpy)
  - [4. `prunesparse.py`](#4-prunesparsepy)
  - [5. `Convert to TensorRT Engine`](#5-convert-to-tensorrt-engine)
  - [6. `Inference with TensorRT Engine`](#6-inference-with-tensorrt-engine)

## Prerequisites

### Environment

Virtual Environment setup for High-resource environment

- Python 3.9.21
- PyTorch-GPU 2.6.0
- CUDA 12.6
- NVIDIA TensorRT 10.10.0.31
- Wandb 0.19.7 (Optional but recommended. For logging purposes)

You can use the following command to install all prerequisites via Conda:

```bash
conda env create -f environment.yml
```

Alternatively, you can use pip with the provided (KIV) `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dataset Setup

This project does not include the dataset files directly. You must download them and place them in the `dataset/` directory.

- **ISCXVPN2016** is available at: [http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/](http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/)
- **Malaya-GT** is available via: TBA

The expected structure is:

```
LiteNet/
├── dataset/
│   ├── ISCXVPN2016/
│   │   ├── train.npy
│   │   ├── test.npy
│   │   └── val.npy
```

## Project Pipeline

The project follows these main steps:

1.  **Configuration (`config.yaml`)**: Central configuration file for all parameters.
2.  **Feature Selection (`feature_selection.py`)**: Identifies the most important features from the dataset.
3.  **Training (`main.py`)**: Trains the `LiteNet` model using the selected features.
4.  **Optimization (`prunesparse.py`)**: Applies 2:4 semi-structured pruning, fine-tunes, and quantizes the model, exporting to ONNX.
5.  **TensorRT Conversion**: The optimized ONNX model is converted into a TensorRT engine.
6.  **Inference (`tensorrtinference.py`)**: The final TensorRT engine is used for high-performance inference and benchmarking.

## How to Use

This section details the usage of each key script in the pipeline.

### 1. `config.yaml`

This file is the central hub for configuring the entire pipeline. It contains settings for:

- **Base Parameters**: Learning rate, batch size, number of epochs.
- **Model Architecture**: Input sequence length, number of features.
- **Dataset specifics**: Class names, number of classes, and the path to the feature list file for each dataset.
- **Feature Selection**: Configuration for the larger model used during the feature selection process.

Before running any scripts, you should review this file to ensure the parameters match your desired configuration and dataset.

### 2. `feature_selection.py`

This script identifies the most important features from your dataset. It trains a `LiteNetLarge` model with the full feature set and then uses SHAP (SHapley Additive exPlanations) to calculate the importance of each feature. The script generates a `.npy` file with the list of the top 20 most important feature indices.

**Modes:**

- `--mode tr`: **Train & Select.** Trains the model from scratch and then runs the SHAP analysis.
- `--mode fs`: **Feature Select only.** Skips training and uses a pre-existing model to run the SHAP analysis.

**Usage:**
To run training followed by feature selection on the `ISCXVPN2016` dataset, use:

```bash
python feature_selection.py --data ISCXVPN2016 --mode tr
```

### 3. `main.py`

After identifying the most important features, this script is used to train the `LiteNet` model using only selected features from `top_features_<dataset>.npy`(specified in `config.yaml`) and preprocesses the data accordingly. It handles both training a new model from scratch and testing a pre-trained model.

**Modes:**

- **Training Mode (default):** Trains, validates, and then tests the model. A trained model (`.pth`) will be saved.
- `--test True`: **Test-Only Mode.** Skips the training process and directly evaluates a pre-trained model on the test dataset. Default file is LiteNet\_<dataset>\_embedding.pth. Use the --path flag to override the default file

**Usage:**

To train the model on the `ISCXVPN2016` dataset with the selected features:

```bash
python main.py --dataset_name ISCXVPN2016
```

To test a pre-trained model:

```bash
python main.py --dataset_name ISCXVPN2016 --test True --path <name_of_your_model>.pth
```

### 4. `prunesparse.py`

This script is responsible for optimizing the trained `LiteNet` model. The process involves three main stages:

1.  **Pruning:** Applies 2:4 semi-structured sparsity to the model. By default, this is only applied to the `Linear` layers due to the lightweight design of the architecture.
2.  **Fine-tuning:** After pruning, the model is fine-tuned for a few epochs to recover any accuracy lost during pruning. The fine-tuning hyperparameters (`fine_tune_epochs`, `fine_tune_lr`) are specified in `config.yaml`.
3.  **Quantization & Export:** The fine-tuned model is then optionally quantized and finally exported to the ONNX format.

**Flags:**

- `--quantization [None|FP16|INT8]`: Specifies the quantization to apply after fine-tuning. Defaults to `None`.
- `--quantize-only`: Skips the pruning and fine-tuning steps, loading a pre-existing fine-tuned model to perform quantization and export.

**Usage:**

To run the full prune, fine-tune, and FP16 quantization pipeline:

```bash
python prunesparse.py --dataset_name ISCXVPN2016 --quantization FP16
```

To run quantization only on an already fine-tuned model:

```bash
python prunesparse.py --dataset_name ISCXVPN2016 --quantization FP16 --quantize-only --path LiteNet_ISCXVPN2016_pruned_finetuned_embedding.pth
```

**Note on ONNX Export:** The script exports the final model to an `.onnx` file. Ensure that model's input shape in the export script matches the subsequent conversion to a TensorRT engine.

### 5. `Convert to TensorRT Engine`

After `prunesparse.py` generates an optimized ONNX model, you can convert it to a TensorRT engine using the `trtexec` command-line tool. This step builds an engine that is optimized for your specific GPU architecture. For CPU, ONNX is recommended.

### For Linux (Bash)

This script is an example for building a TensorRT inference engine with FP16 quantization for the `ISCXVPN2016` LiteNet model.

**Note:** This TensorRT engine was built on an NVIDIA RTX 4080 Super with TensorRT version 10.10.0.31. The engine should be created on the deployed device or it will not work properly.
**Note:** The `--shapes` flag and `INPUT_NAME` must be the same as the `.onnx` model's input. This line is written for a fixed batch size scenario.

```bash
DATASET="ISCXVPN2016" # Specify dataset
QUANT="FP16" # Specify quantization
ONNX_MODEL_DIR="saved_dict"
ONNX_MODEL_PATH="${ONNX_MODEL_DIR}/LiteNet_${DATASET}_pruned_finetuned_embedding_${QUANT}.onnx"
TRT_ENGINE_PATH="${ONNX_MODEL_DIR}/LiteNet_${DATASET}_pruned_finetuned_embedding_${QUANT}.trt"
INPUT_NAME="input" # onnx input name
BATCH_SIZE=64
FEATURES=20

trtexec --onnx=${ONNX_MODEL_PATH} \
        --saveEngine=${TRT_ENGINE_PATH} \
 	--sparsity=enable \
        --useCudaGraph \
 	--shapes=${INPUT_NAME}:${BATCH_SIZE}x${FEATURES} \
	--fp16
```

## Inference with TensorRT Engine

### 5. `tensorrtinference.py`

This script is used to run and benchmark the final `.trt` inference engine. It loads the engine, performs 100 warmup runs to ensure stable GPU performance, and then evaluates the engine on the entire test dataset, reporting metrics like throughput (QPS), latency, and accuracy.

**Flags:**

- `--data`: Specifies the dataset used for inference. This determines which test data to load.
- `--quantization`: Specifies the precision of the TensorRT engine (`FP16` or `INT8`) to ensure the correct engine file is loaded.
- `--path`: (Optional) Allows specifying a direct path to the `.trt` engine file, overriding the default name.

**Usage:**

To benchmark an `FP16` engine for the `ISCXVPN2016` dataset:

```bash
python tensorrtinference.py --data ISCXVPN2016 --quantization FP16
```
