import pathlib

import numpy as np
import pandas as pd

import os
import sys

src_path = os.path.dirname(os.path.abspath("./"))
print(src_path)
sys.path.append(src_path)

from commonpath import LABELLED_DATA_DIR

DIR_PATH = pathlib.Path("./chunks_output")
CLEAN_PATH = pathlib.Path("./chunks_output_clean")


def clean_files():
    CLEAN_PATH.mkdir(exist_ok=True)

    for fp in DIR_PATH.glob("*.csv"):
        with open(fp, "r") as f:
            content = f.readlines()
        with open(CLEAN_PATH / fp.name, "w") as f:
            f.writelines(content[1:-1])


def read_and_verify():
    for fp in CLEAN_PATH.glob("*.csv"):
        print(fp)
        if fp.name.endswith("251.csv"):
            dst_length = 22
        else:
            dst_length = 50
        data = pd.read_csv(fp)
        if len(data) != dst_length:
            print("len(data) != dst_length:", len(data), dst_length)


def merge_csv():
    df = None
    for fp in CLEAN_PATH.glob("*.csv"):
        try:
            data = pd.read_csv(fp)
        except pd.errors.ParserError:
            continue
        if df is None:
            df = data
        else:
            df = pd.concat([df, data])
    print(df)
    df = df[~df["ID"].isna()]
    df["ID"] = df["ID"].apply(int)
    mask1 = np.logical_and(df["Is Binding"], ~df["Source Library"].isna())
    mask2 = df.apply(
        lambda x: x["Source Library"] in x["Description"] if isinstance(x["Source Library"], str) and isinstance(
            x["Description"], str) else False, axis=1)
    mask = np.logical_and(mask1, mask2)
    df.loc[mask, "Is Binding"] = True
    df.loc[np.logical_not(mask), "Is Binding"] = False
    df["WebJar"] = False
    df.loc[mask, "WebJar"] = df.loc[mask, "Description"].apply(lambda x: "WebJar" in x if isinstance(x, str) else False)

    df.to_csv("labelled_cross_eco.csv", index=False)
    label_true = df[mask]

    def has_indicator(description):
        # Check for common binding/wrapper indicators in name or description
        indicators = ['binding', 'wrap', 'bridge', 'interface', 'connector', "webjar", "api", "sdk"]
        for indicator in indicators:
            if indicator in description.lower():
                return True
        return False

    label_true_mask_1 = np.logical_and(label_true["Is Binding"], np.logical_not(label_true["WebJar"]))
    label_true_mask_2 = label_true["Description"].apply(has_indicator)
    label_true_mask = np.logical_and(label_true_mask_1, label_true_mask_2)
    # label_true_mask = label_true_mask.sample(frac=1, random_state=123, ignore_index=True)
    label_true_saved = label_true[label_true_mask].reset_index().copy()

    all_repos = pd.read_csv("cross_ecosystem_packages_by_repo.csv").reset_index()
    all_repos = all_repos[["ID", "Repository Stars Count"]]
    label_true_saved = label_true_saved[["ID", "Platform", "Name", "Description", "Is Binding", "Source Library"]]
    label_true_saved = label_true_saved.merge(all_repos, on="ID")
    label_true_saved = label_true_saved.sort_values("Repository Stars Count", ascending=False)

    # label_true_saved = label_true_saved.reset_index()
    label_true_saved[
        ["ID", "Platform", "Name", "Description", "Is Binding", "Source Library", "Repository Stars Count"]
    ].to_csv("labelled_true_sorted.csv", index=False)

    # label_true_saved = label_true_saved.sample(frac=1, random_state=123, ignore_index=True)
    # label_true_saved[["ID", "Platform", "Name", "Description", "Is Binding", "Source Library"]].to_csv("labelled_true.csv", index=False)


def clean_csv():
    DATA_PATH = pathlib.Path("./data_clean")
    labelled_true = pd.read_csv(DATA_PATH / "labelled_true.csv")
    mask = labelled_true["Is Binding"] == "TRUE"
    saved_data = labelled_true[mask].copy()
    saved_data.to_csv(DATA_PATH / "added_labelled_data_source.csv")
    saved_data["id"] = saved_data["ID"].astype(int)
    saved_data["context"] = saved_data["Description"]
    saved_data["answer"] = saved_data["Source Library"]
    def get_answer_positions(context, answer):
        start_pos = context.index(answer)
        end_pos = start_pos+len(answer)
        print(context, answer, "====", context[start_pos:end_pos])
        return start_pos, end_pos

    saved_data["answer_start"] = saved_data.apply(lambda x: get_answer_positions(x["context"], x["answer"])[0], axis=1)
    saved_data["answer_end"] = saved_data.apply(lambda x: get_answer_positions(x["context"], x["answer"])[1], axis=1)
    saved_data["question"] = "What is the source library of this binding?"
    saved_data["is binding"] = "True"
    saved_data["diff"] = "False"
    saved_data[[
        "id", "answer_start", "answer", "context", "question", "answer_end", "is binding", "diff"
    ]].to_csv(DATA_PATH / "added_labelled_data.csv", index=False)


def merge_csv_and_random():
    RANDOM_STATE = 23
    DATA_PATH = pathlib.Path("./data_clean")
    df1 = pd.read_csv(DATA_PATH / "added_labelled_data.csv")
    df2 = pd.read_csv(DATA_PATH / "updated_labelled_QA_2437samples.csv")
    df = pd.concat([df1, df2])
    df.dropna(subset="context", inplace=True)
    df.drop_duplicates("id", inplace=True)

    df["answer"] = df.apply(lambda x: "" if x["answer_start"] == -1 else x["answer"], axis=1)
    df["is binding"] = df.apply(lambda x: False if x["answer_start"] == -1 else x["is binding"], axis=1)
    verify_df(df)
    #
    # assert df["id"].dtype == int
    # assert df["answer_start"].dtype == int
    # assert df["answer_end"].dtype == int
    # # assert df["answer"].dtype == str
    # # assert df["context"].dtype == str
    # # assert df["question"].dtype == str
    # assert df["is binding"].dtype == bool
    #
    # def is_answer_correct(x):
    #     if x["answer_start"] == -1:
    #         return True
    #     return x["answer"] == x["context"][x["answer_start"]:x["answer_end"]]
    # is_all_correct = df.apply(is_answer_correct, axis=1)
    # assert all(is_all_correct)
    #
    # def is_label_consistent(x):
    #     not_binding_mask1 = x["answer_start"] == -1
    #     not_binding_mask2 = x["is binding"] == False
    #     return not_binding_mask1 == not_binding_mask2
    # is_all_label_consistent = df.apply(is_label_consistent, axis=1)
    # print(df[~is_all_label_consistent])
    # df.to_csv(DATA_PATH / "labelled_data.csv", index=False)
    # assert all(is_all_label_consistent)
    # # df = df.sample(frac=1, random_state=123, ignore_index=True)
    # print(df)


def verify_df(df):
    def is_answer_correct(x):
        if x["answer_start"] == -1:
            return True
        return x["answer"] == x["context"][x["answer_start"]:x["answer_end"]]
    is_all_correct = df.apply(is_answer_correct, axis=1)
    print(df[~is_all_correct])
    assert all(is_all_correct)

    def is_label_consistent(x):
        not_binding_mask1 = x["answer_start"] == -1
        not_binding_mask2 = x["is binding"] == False
        return not_binding_mask1 == not_binding_mask2
    is_all_label_consistent = df.apply(is_label_consistent, axis=1)
    print(df[~is_all_label_consistent])
    assert all(is_all_label_consistent)
    # df = df.sample(frac=1, random_state=123, ignore_index=True)
    print(df)


def split_dataset(random_state=42):
    SAMPLE_NUM = 25
    data_path = LABELLED_DATA_DIR
    df = pd.read_csv(data_path / "labelled_data.csv")
    print(len(df))
    df.drop_duplicates("context", inplace=True)
    print(len(df))
    df.drop("diff", axis=1, inplace=True, errors="ignore")
    print(len(df))
    df["answer"] = df.apply(lambda x: "" if x["answer_start"] == -1 else x["answer"], axis=1)
    df["is binding"] = df.apply(lambda x: False if x["answer_start"] == -1 else x["is binding"], axis=1)
    verify_df(df)
    df.to_csv(data_path / "labelled_data_clean.csv", index=False)

    print(df["is binding"].value_counts())

    # Separate the DataFrame into two based on the label
    df_true = df[df['is binding'] == True]
    df_false = df[df['is binding'] == False]

    # Random sampling for validation set
    validation_true = df_true.sample(SAMPLE_NUM, random_state=random_state)
    validation_false = df_false.sample(SAMPLE_NUM, random_state=random_state)

    # Remove the validation samples from the original datasets
    df_true = df_true.drop(validation_true.index)
    df_false = df_false.drop(validation_false.index)

    # Random sampling for test set
    test_true = df_true.sample(SAMPLE_NUM, random_state=random_state)
    test_false = df_false.sample(SAMPLE_NUM, random_state=random_state)

    # Remove the test samples from the original datasets
    df_true = df_true.drop(test_true.index)
    df_false = df_false.drop(test_false.index)

    # Concatenate to form the validation, test, and training sets
    validation_set = pd.concat([validation_true, validation_false])
    test_set = pd.concat([test_true, test_false])
    training_set = pd.concat([df_true, df_false])

    # Shuffle the sets
    validation_set = validation_set.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=random_state).reset_index(drop=True)
    training_set = training_set.sample(frac=1, random_state=random_state).reset_index(drop=True)

    training_set.to_csv(data_path / "binding_QA_train.csv", index=False)
    validation_set.to_csv(data_path / "binding_QA_validation.csv", index=False)
    test_set.to_csv(data_path / "binding_QA_test.csv", index=False)


if __name__ == '__main__':
    # clean_files()
    # read_and_verify()
    # merge_csv()
    # clean_csv()
    # merge_csv_and_random()

    split_dataset()
