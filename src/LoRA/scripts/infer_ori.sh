#!/bin/sh
# medalpaca prompt 1
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --use_lora False \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name 'medalpaca_1.csv' \
    --prompt_id '1' \
    --batch_size 4 \
    --num_beams 1 \
    --max_new_tokens 64
# medalpaca prompt 4
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --use_lora False \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name 'medalpaca_4.csv' \
    --prompt_id '4' \
    --batch_size 2 \
    --num_beams 4 \
    --max_new_tokens 256