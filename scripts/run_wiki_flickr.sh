#!/bin/bash

# Train models on wiki+flickr dataset.
# For SimCSE baseline, you just need to (1) set new output_dir (2) --framework simcse (3) remove --feature_file

IMG=data/flickr_resnet.hdf5
CAPTION=data/flickr_random_captions.txt
TEXT=data/wiki1m_for_simcse.txt

SEED=0
MODEL=bert-base-uncased
LR=3e-5
BATCH=64
EPS=3
LBD=0.01

OUT_DIR=result/mix_flickr/${SEED}/mcse

python src/train_mix.py \
    --framework mcse \
    --model_name_or_path $MODEL \
    --text_file $TEXT \
    --caption_file $CAPTION  \
    --feature_file $IMG \
    --output_dir $OUT_DIR \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH \
    --num_train_epochs $EPS \
    --seed $SEED  \
    --lbd $LBD


python simcse_to_huggingface.py --path $OUT_DIR

python src/evaluation.py \
      --model_name_or_path $OUT_DIR \
      --pooler cls_before_pooler \
      --task_set sts \
      --mode test
