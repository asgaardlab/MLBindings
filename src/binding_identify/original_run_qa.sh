#!/bin/bash

rm qa_finished

# Define common variables
export DATA_DIR="../../data/labelled_data/"
export OUTPUT_DIR="../../output/binding_qa"

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED="true"
export METRIC_NAME="f1"
export STEPS=15
export TRAIN_FILE="$DATA_DIR/binding_QA_train.csv"
export VALIDATION_FILE="$DATA_DIR/binding_QA_validation.csv"
export TEST_FILE="$DATA_DIR/binding_QA_test.csv"
export MAX_SEQ_LENGTH=384
export EVAL_STRATEGY="steps"
export SAVE_STRATEGY="steps"
export SAVE_TOTAL_LIMIT=1
export LOAD_BEST_MODEL_AT_END=True
export LEARNING_RATE=3e-5
export DOC_STRIDE=0

export DEFAULT_TRAIN_EPOCH=10
export DEFAULT_BATCH_SIZE=32

# Define a dictionary of model names, batch sizes, and TRAIN_EPOCH values
declare -A model_settings
# model_settings["albert-base-v2"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"
model_settings["albert-base-v1"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"
# model_settings["roberta-base"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"
# model_settings["distilbert-base-uncased"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"
# model_settings["distilbert-base-cased"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"
# model_settings["bert-base-uncased"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"
# model_settings["bert-base-cased"]="$DEFAULT_BATCH_SIZE $DEFAULT_TRAIN_EPOCH"

# Loop through the models and their settings in the dictionary
for MODEL_NAME in "${!model_settings[@]}"
do
    SETTINGS=(${model_settings[$MODEL_NAME]})
    BATCH_SIZE="${SETTINGS[0]}"
    TRAIN_EPOCH="${SETTINGS[1]}"

    python original_run_qa.py \
      --model_name_or_path $MODEL_NAME \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --test_file $TEST_FILE \
      --do_train \
      --do_eval \
      --do_predict \
      --version_2_with_negative \
      --per_device_train_batch_size $BATCH_SIZE \
      --learning_rate $LEARNING_RATE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --num_train_epochs $TRAIN_EPOCH \
      --max_seq_length $MAX_SEQ_LENGTH \
      --evaluation_strategy $EVAL_STRATEGY \
      --save_strategy $SAVE_STRATEGY \
      --logging_steps $STEPS \
      --pad_to_max_length \
      --eval_steps $STEPS \
      --save_steps $STEPS \
      --doc_stride $DOC_STRIDE \
      --max_answer_length 100 \
      --metric_for_best_model $METRIC_NAME \
      --save_total_limit $SAVE_TOTAL_LIMIT \
      --load_best_model_at_end $LOAD_BEST_MODEL_AT_END \
      --overwrite_output_dir \
      --output_dir $OUTPUT_DIR/$MODEL_NAME/
done

# touch qa_finished


export STEPS=30
unset model_settings
declare -A model_settings
# model_settings["albert-large-v2"]="16 $DEFAULT_TRAIN_EPOCH"
# model_settings["albert-xlarge-v2"]="8 $DEFAULT_TRAIN_EPOCH"
# model_settings["albert-xxlarge-v2"]="8 $DEFAULT_TRAIN_EPOCH"
model_settings["albert-large-v1"]="16 $DEFAULT_TRAIN_EPOCH"
model_settings["albert-xlarge-v1"]="8 $DEFAULT_TRAIN_EPOCH"
model_settings["albert-xxlarge-v1"]="8 $DEFAULT_TRAIN_EPOCH"
# model_settings["roberta-large"]="16 $DEFAULT_TRAIN_EPOCH"
# model_settings["deepset/roberta-base-squad2"]="16 $DEFAULT_TRAIN_EPOCH"
# model_settings["deepset/roberta-large-squad2"]="16 $DEFAULT_TRAIN_EPOCH"
# model_settings["bert-large-uncased"]="16 $DEFAULT_TRAIN_EPOCH"
# model_settings["bert-large-cased"]="16 $DEFAULT_TRAIN_EPOCH"
# model_settings["facebook/bart-base"]="16 $DEFAULT_TRAIN_EPOCH"

# Loop through the models and their settings in the dictionary
for MODEL_NAME in "${!model_settings[@]}"
do
    SETTINGS=(${model_settings[$MODEL_NAME]})
    BATCH_SIZE="${SETTINGS[0]}"
    TRAIN_EPOCH="${SETTINGS[1]}"

    python original_run_qa.py \
      --model_name_or_path $MODEL_NAME \
      --train_file $TRAIN_FILE \
      --validation_file $VALIDATION_FILE \
      --test_file $TEST_FILE \
      --do_train \
      --do_eval \
      --do_predict \
      --version_2_with_negative \
      --per_device_train_batch_size $BATCH_SIZE \
      --learning_rate $LEARNING_RATE \
      --per_device_eval_batch_size $BATCH_SIZE \
      --num_train_epochs $TRAIN_EPOCH \
      --max_seq_length $MAX_SEQ_LENGTH \
      --evaluation_strategy $EVAL_STRATEGY \
      --save_strategy $SAVE_STRATEGY \
      --logging_steps $STEPS \
      --pad_to_max_length \
      --eval_steps $STEPS \
      --save_steps $STEPS \
      --doc_stride $DOC_STRIDE \
      --max_answer_length 100 \
      --metric_for_best_model $METRIC_NAME \
      --save_total_limit $SAVE_TOTAL_LIMIT \
      --load_best_model_at_end $LOAD_BEST_MODEL_AT_END \
      --overwrite_output_dir \
      --output_dir $OUTPUT_DIR/$MODEL_NAME/
done


#
#export CUDA_VISIBLE_DEVICES=0
#export WANDB_DISABLED="true"
#
#python original_run_qa.py \
#  --model_name_or_path albert-base-v2 \
#  --dataset_name ./updated_labelled_QA_2437samples.csv \
#  --do_train \
#  --do_eval \
#  --do_predict \
#  --version_2_with_negative \
#  --per_device_train_batch_size 12 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 5 \
#  --max_seq_length 384 \
#  --doc_stride 128 \
#  --max_answer_length 100 \
#  --evaluation_strategy "steps" \
#  --save_strategy "steps" \
#  --logging_steps 10 \
#  --eval_steps 10 \
#  --save_steps 10 \
#  --metric_for_best_model "HasAns_f1" \
#  --save_total_limit 1 \
#  --load_best_model_at_end True \
#  --overwrite_output_dir \
#  --output_dir ./original_new_util/albert-base-v2
