PRE_SEQ_LEN=128
CHECKPOINT=0523-bio_prompt_1-chatglm-6b-pt-128-2e-2-bs8-accumulation2
STEP=34900

CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --do_predict \
    --validation_file ../../data/val_prompt.json \
    --test_file ../../data/test_prompt.json \
    --overwrite_cache \
    --prompt_column prompt \
    --response_column completion \
    --model_name_or_path /home/shinian.ljl/projects/ChatGLM-6B/THUDM/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN
