#!/bin/bash

conda activate binding_qa
export CUDA_VISIBLE_DEVICES=1

# # The file we are waiting for
# file="finished"

# # Loop until the file exists
# while [ ! -f "$file" ]; do
#   echo "Waiting for file '$file' to be created..."
#   sleep 60 # Wait for 1 second before checking again
# done

# echo "File '$file' has been created."

export MAX_SEQ_LENGTH=384
export DEFAULT_BATCH_SIZE=32
export METRIC_NAME="f1"

# Define a dictionary of model names with their respective batch sizes
declare -A model_settings

model_settings["albert-large-v2"]="16"
# model_settings["albert-xlarge-v2"]="8"
model_settings["albert-xxlarge-v2"]="8"
model_settings["roberta-large"]="16"
model_settings["deepset/roberta-base-squad2"]="16"
model_settings["bert-large-uncased"]="16"
model_settings["bert-large-cased"]="16"
model_settings["facebook/bart-base"]="16"

model_settings["albert-base-v2"]="$DEFAULT_BATCH_SIZE"
model_settings["roberta-base"]="$DEFAULT_BATCH_SIZE"
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
        output_dir="./binding_classification_infer/${model}/out_$test_file"

        # Run the command
        python binding_classification.py --model_name_or_path "./binding_classification/$model" \
                         --dataset_name ./updated_labelled_QA_2437samples.csv \
                         --test_file ./all_data/$test_file \
                         --do_predict \
                         --per_device_eval_batch_size $batch_size \
                         --max_seq_length $MAX_SEQ_LENGTH \
                         --metric_for_best_model $METRIC_NAME \
                         --output_dir $output_dir
    done
done
