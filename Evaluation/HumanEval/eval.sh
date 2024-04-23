MODEL_NAME_OR_PATH=$1
DATASET_ROOT="data/"
LANGUAGE="python"
CUDA_VISIBLE_DEVICES=6 python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}
