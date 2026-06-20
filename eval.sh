#!/bin/bash

export MASTER_PORT="${MASTER_PORT:-29509}"

# GPU/worker settings aligned with the training scripts.
DEVICE="${DEVICE:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MIN_PIXELS="${MIN_PIXELS:-262144}"
MAX_PIXELS="${MAX_PIXELS:-262144}"

# Pass dataset directories as arguments, or override DATA_DIR.
if [ "$#" -gt 0 ]; then
    DATA_DIRS=("$@")
else
    DATA_DIRS=("${DATA_DIR:-DATA/Test}")
fi

RESULT_DIR="${RESULT_DIR:-output}"
MODEL_PATH="${MODEL_PATH:-Checkpoints_Stage1}"
REGISTER_FILE="${REGISTER_FILE:-Models/DeepVRM/registers_file.py}"
OPTIM_FILE="${OPTIM_FILE:-Models/DeepVRM/custom_optim.py}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-my_qwen2_5_vl}"

# Set ADAPTER_PATH to evaluate a LoRA/custom adapter checkpoint.
ADAPTER_PATH="${ADAPTER_PATH:-}"

shopt -s nullglob

for current_data_dir in "${DATA_DIRS[@]}"; do
    sub_dir_name=$(basename "$current_data_dir")
    current_result_dir="$RESULT_DIR/$sub_dir_name"
    mkdir -p "$current_result_dir"

    TEST_DATASETS=("$current_data_dir"/*.json)

    echo "Processing directory: $sub_dir_name (${#TEST_DATASETS[@]} JSON files found)"
    echo "=========================================="

    for i in "${!TEST_DATASETS[@]}"; do
        input_file="${TEST_DATASETS[$i]}"
        filename=$(basename "$input_file")
        dataset_name=$(basename "$filename" ".json")

        echo "[$((i + 1))/${#TEST_DATASETS[@]}] Evaluating: $dataset_name"

        output_file="$current_result_dir/${dataset_name}_test_result.jsonl"

        if [ -f "$output_file" ]; then
            echo "Output file $output_file already exists, skipping..."
            continue
        fi

        if [ -z "$ADAPTER_PATH" ]; then
            NPROC_PER_NODE="$NPROC_PER_NODE" \
            CUDA_VISIBLE_DEVICES="$DEVICE" \
            MIN_PIXELS="$MIN_PIXELS" \
            MAX_PIXELS="$MAX_PIXELS" \
            swift infer \
                --model_type "$TEMPLATE_TYPE" \
                --template "$TEMPLATE_TYPE" \
                --val_dataset "$input_file" \
                --torch_dtype 'bfloat16' \
                --infer_backend pt \
                --model "$MODEL_PATH" \
                --result_path "$output_file" \
                --attn_impl flash_attn \
                --max_new_tokens 64 \
                --max_model_len 4096 \
                --custom_register_path "$REGISTER_FILE" \
                --external_plugins "$OPTIM_FILE"
        else
            NPROC_PER_NODE="$NPROC_PER_NODE" \
            CUDA_VISIBLE_DEVICES="$DEVICE" \
            MIN_PIXELS="$MIN_PIXELS" \
            MAX_PIXELS="$MAX_PIXELS" \
            swift infer \
                --model_type "$TEMPLATE_TYPE" \
                --template "$TEMPLATE_TYPE" \
                --val_dataset "$input_file" \
                --torch_dtype 'bfloat16' \
                --infer_backend pt \
                --model "$MODEL_PATH" \
                --result_path "$output_file" \
                --max_new_tokens 64 \
                --attn_impl flash_attn \
                --max_model_len 4096 \
                --adapters "$ADAPTER_PATH" \
                --custom_register_path "$REGISTER_FILE" \
                --external_plugins "$OPTIM_FILE"
        fi

        if [ $? -eq 0 ]; then
            echo "$dataset_name completed"
        else
            echo "$dataset_name failed"
        fi
    done
done

echo "=========================================="
echo "All evaluation tasks completed - $(date)"
