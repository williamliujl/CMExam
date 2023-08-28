#!/bin/bash
prompt_template="med_template"
prompt_id="1"
num_epochs=10
# LLaMA-CMExam
exp_tag="LLaMA-CMExam"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
# Alpaca-CMExam
exp_tag="Alpaca-CMExam"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --resume_from_checkpoint 'alpaca-lora-7b' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'
# Huatuo-CMExam
exp_tag="Huatuo-CMExam"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --resume_from_checkpoint 'lora-alpaca-med' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
# MedAlpaca-CMExam
exp_tag="Medalpaca-CMExam"
python finetune.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
#
prompt_id="4"
num_epochs=1
# LLaMA-CMExam
exp_tag="LLaMA-CMExam"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
# Alpaca-CMExam
exp_tag="Alpaca-CMExam"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --resume_from_checkpoint 'alpaca-lora-7b' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'
# Huatuo-CMExam
exp_tag="Huatuo-CMExam"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --resume_from_checkpoint 'lora-alpaca-med' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
# MedAlpaca-CMExam
exp_tag="Medalpaca-CMExam"
python finetune.py \
    --base_model 'medalpaca/medalpaca-7b' \
    --data_path '../../data/train_prompt.json' \
    --valid_data_path '../../data/val_prompt.json' \
    --output_dir './saved/lora-cmexam-'${exp_tag}'-'${prompt_id} \
    --prompt_template_name $prompt_template \
    --micro_batch_size 8 \
    --batch_size 128 \
    --wandb_run_name $exp_tag \
    --prompt_id $prompt_id \
    --num_epochs $num_epochs \
    --cutoff_len 256 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16