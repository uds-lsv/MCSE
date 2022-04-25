#!/bin/bash

# Train models on wiki1m dataset.

TEXT=data/wiki1m_for_simcse.txt

SEED=0
MODEL=bert-base-uncased
LR=3e-5
BATCH=64
EPS=3

OUT_DIR=result/wiki1m/${SEED}/simcse

python src/train.py \
    --framework simcse \
    --model_name_or_path  $MODEL\
    --text_file $TEXT \
    --output_dir $OUT_DIR \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH \
    --num_train_epochs $EPS  \
    --seed $SEED

python simcse_to_huggingface.py --path $OUT_DIR

python src/evaluation.py \
        --model_name_or_path $OUT_DIR \
        --pooler cls_before_pooler \
        --task_set sts \
        --mode test
