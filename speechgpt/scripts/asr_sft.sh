#!/bin/bash

# METAROOT="output/stage2/"
METAROOT="llama/3_2/3B/Llama-3.2-3B-Instruct"
DATAROOT="data/stage2"
OUTROOT="output/stage2"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/


#ddp realted
# NNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# NODE_RANK=$(($(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -Fn $(hostname) | cut --delimiter=":" --fields=1)-1))


echo "stage2: asr fine-tuning"


# torchrun \
#     --nnode $NNODE \
#     --nproc_per_node 8 \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port 29501  \

# --nproc_per_node를 사용할 gpu 개수로 설정하면 됨
torchrun \
    --nproc_per_node 2 \
    --standalone \
src/train/asr_sft.py \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/asr_data_librispeech_styletalk.jsonl" \
    --val_set_size 29 \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 10 \
    --model_max_length 2048 \
    --bf16 True \
    --do_train \
    --do_eval \
    --train_on_inputs True \
    --output_dir "${OUTROOT}" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --overwrite_output_dir \
    --train_embeddings \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

