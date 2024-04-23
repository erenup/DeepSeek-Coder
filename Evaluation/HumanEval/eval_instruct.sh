LANG="python"
OUPUT_DIR="output"
#MODEL="../../finetune/test_code/checkpoint-100"
MODEL=$1

CUDA_VISIBLE_DEVICES=7 python eval_instruct.py \
    --model "$MODEL" \
    --output_path "$MODEL/output.jsonl" \
    --language $LANG \
    --temp_dir $OUPUT_DIR
