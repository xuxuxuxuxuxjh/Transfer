#!/bin/bash

source ~/.bashrc

eval "$(/root/miniconda3/bin/conda shell.bash hook)"

conda activate unsloth_env

python GRPO.py >logs/output_Huatuo_GRPO2.log 

# accelerate launch --multi_gpu --num_processes 4 --main_process_port=29504 GRPO_Full.py >logs/output_GRPO_Full.log

# deepspeed --num_gpus 4 --master_port 29501 Baichuan.py --deepspeed zero3.json >logs/output_Baichuan.log

# torchrun --nproc_per_node=4 --master_port=29508 \
# --nnodes=1 --node_rank=0 \
# GRPO_Full.py >logs/output_GRPO_Full.log