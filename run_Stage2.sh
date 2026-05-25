
DATASETS="Your Dataset"

export MASTER_PORT=29502

# # ---------------------
OMP_THREAD_NUM=1 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MIN_PIXELS=262144 \
MAX_PIXELS=262144 \
swift sft \
    --model Checkpoints_Stage1 \
    --template 'my_qwen2_5_vl' \
    --model_type 'my_qwen2_5_vl' \
    --dataset $DATASETS \
    --torch_dtype 'bfloat16' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --aligner_lr 1e-6 \
    --weight_decay 1e-3 \
    --gradient_accumulation_steps 2 \
    --eval_steps 350 \
    --save_steps 350 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir 'Checkpoints/Stage2_first_mid_feat_vit_llm_lora/lora_r64_a128/' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --train_type custom \
    --lora_rank 64 \
    --lora_alpha 128 \
    --external_plugins "Models/DeepVRM/custom_optim.py" \
    --custom_register_path "Models/DeepVRM/registers_file.py" \
    --split_dataset_ratio 0.05 \
    --freeze_aligner false \
    --freeze_vit false \
    --freeze_llm false \
    --enable_channel_loss true \
    --attn_impl flash_attn
