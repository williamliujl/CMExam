PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --do_train \
    --train_file /home/shinian.ljl/data/bio/CMedQA/train_prompt_1.json \
    --validation_file /home/shinian.ljl/data/bio/CMedQA/val_prompt_1.json \
    --prompt_column prompt \
    --response_column completion \
    --overwrite_cache \
    --model_name_or_path /home/shinian.ljl/projects/ChatGLM-6B/THUDM/chatglm-6b \
    --output_dir output/0813-bio_prompt_1-chatglm-6b-pt-$PRE_SEQ_LEN-$LR-bs8-accumulation2 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 50000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --report_to wandb