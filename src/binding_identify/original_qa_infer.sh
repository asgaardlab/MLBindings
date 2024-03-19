#!/bin/bash

conda activate binding_qa
export CUDA_VISIBLE_DEVICES=1

# # The file we are waiting for
# file="qa_finished"

# # Loop until the file exists
# while [ ! -f "$file" ]; do
#   echo "Waiting for file '$file' to be created..."
#   sleep 60 # Wait for 1 second before checking again
# done

# echo "File '$file' has been created."

export DATA_DIR="../../data/project_repo_chunks_for_qa"
export MODEL_DIR="../../output/binding_qa_strict"
export OUTPUT_DIR="../../output/binding_qa_strict_infer"

export METRIC_NAME="f1"
export STEPS=15
export MAX_SEQ_LENGTH=384
export EVAL_STRATEGY="steps"
export SAVE_STRATEGY="steps"
export SAVE_TOTAL_LIMIT=1
export LOAD_BEST_MODEL_AT_END=True
export LEARNING_RATE=3e-5
export DOC_STRIDE=0

export DEFAULT_BATCH_SIZE=32

# Define a dictionary of model names with their respective batch sizes
declare -A model_settings

# model_settings["albert-large-v2"]="16"
# model_settings["albert-xlarge-v2"]="8"
# model_settings["albert-xxlarge-v2"]="8"
# model_settings["roberta-large"]="16"
model_settings["deepset/roberta-base-squad2"]="16"
# model_settings["deepset/roberta-large-squad2"]="16"
# model_settings["bert-large-uncased"]="16"
# model_settings["bert-large-cased"]="16"
# model_settings["facebook/bart-base"]="16"

# model_settings["albert-base-v2"]="$DEFAULT_BATCH_SIZE"
# model_settings["roberta-base"]="$DEFAULT_BATCH_SIZE"
# model_settings["distilbert-base-uncased"]="$DEFAULT_BATCH_SIZE"
# model_settings["distilbert-base-cased"]="$DEFAULT_BATCH_SIZE"
# model_settings["bert-base-uncased"]="$DEFAULT_BATCH_SIZE"
# model_settings["bert-base-cased"]="$DEFAULT_BATCH_SIZE"

# Define the array of test files
test_files=("all_data_0.csv" "all_data_1.csv" "all_data_2.csv" "all_data_3.csv" "all_data_4.csv" "all_data_5.csv" "all_data_6.csv" "all_data_7.csv" "all_data_8.csv" "all_data_9.csv" "all_data_10.csv" "all_data_11.csv" "all_data_12.csv" "all_data_13.csv" "all_data_14.csv")

# Loop over each model and test file
for model in "${!model_settings[@]}"; do
    for test_file in "${test_files[@]}"; do
        batch_size=${model_settings[$model]}

        # Construct the output directory based on the model and test file
        output_dir="$OUTPUT_DIR/${model}/out_$test_file"

        # Run the command
        python original_run_qa.py --model_name_or_path "$MODEL_DIR/$model" \
                         --dataset_name $DATA_DIR/../updated_labelled_QA_2437samples.csv \
                         --test_file $DATA_DIR/$test_file \
                         --do_predict \
                        --version_2_with_negative \
                        --per_device_eval_batch_size $batch_size \
                        --max_seq_length $MAX_SEQ_LENGTH \
                        --pad_to_max_length \
                        --doc_stride $DOC_STRIDE \
                        --max_answer_length 100 \
                        --metric_for_best_model $METRIC_NAME \
                         --output_dir $output_dir
    done
done
