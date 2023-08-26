#!/bin/sh
# LLaMA-CMExam prompt 1
model_name='LLaMA-CMExam'
prompt_id='1'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 1 \
    --max_new_tokens 32
# LLaMA-CMExam prompt 4
model_name='LLaMA-CMExam'
prompt_id='4'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 4 \
    --max_new_tokens 256
# Alpaca-CMExam prompt 1
model_name='Alpaca-CMExam'
prompt_id='1'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 1 \
    --max_new_tokens 32
# Alpaca-CMExam prompt 4
model_name='Alpaca-CMExam'
prompt_id='4'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 4 \
    --max_new_tokens 256
# Huatuo-CMExam prompt 1
model_name='Huatuo-CMExam'
prompt_id='1'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 1 \
    --max_new_tokens 32
# Huatuo-CMExam prompt 4
model_name='Huatuo-CMExam'
prompt_id='4'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 4 \
    --max_new_tokens 256
# Medalpaca-CMExam prompt 1
model_name='Medalpaca-CMExam'
prompt_id='1'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 1 \
    --max_new_tokens 32
# Medalpaca-CMExam prompt 4
model_name='Medalpaca-CMExam'
prompt_id='4'
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --lora_weights './saved/lora-cmexam-'${model_name}'-'${prompt_id}'/' \
    --use_lora True \
    --instruct_dir '../../data/test_prompt.csv' \
    --prompt_template 'med_template' \
    --output_file_name ${model_name}'-'${prompt_id}'.csv' \
    --prompt_id ${prompt_id} \
    --batch_size 4 \
    --num_beams 4 \
    --max_new_tokens 256