#!/usr/bin/env python

from functools import reduce

import numpy as np
import pandas as pd

import commonpath
import helper


def collect_popular_repos():
    saved_path = commonpath.OUTPUT_DIR / "popular_repos.csv"
    first = True
    for idx, chunk in enumerate(helper.readCSV(
            commonpath.LIBRARIES_IO_REPOSITORIES_FIXED_PATH,
            chunksize=100000
    )):
        extracted = chunk[chunk["Stars Count"] >= helper.REPO_STARS_THRESHOLD]
        if len(extracted):
            extracted.to_csv(saved_path, mode='a', header=first, index=False)
            first = False


def collect_ml_repos():
    saved_path = commonpath.OUTPUT_DIR / "ml_repos.csv"
    first = True
    for idx, chunk in enumerate(helper.readCSV(
            commonpath.LIBRARIES_IO_REPOSITORIES_FIXED_PATH,
            chunksize=1000000
    )):
        matching_fields = [
            "Keywords",
            "Description",  # 3
            # "Name with Owner",  # 4
        ]
        masks = []
        for field in matching_fields:
            mask = chunk[field].apply(lambda x: helper.matchMLKeyWords(x) >= 0)
            masks.append(mask)
        ml_package_mask = reduce(lambda a, b: np.logical_or(a, b), masks)
        extracted = chunk[ml_package_mask]
        if len(extracted):
            extracted.to_csv(saved_path, mode='a', header=first, index=False)
            first = False


def filter_out_toy_projects():
    df = pd.read_csv(commonpath.OUTPUT_DIR / "ml_repos.csv")
    print("length before filter:", len(df))

    print("stars:")
    stars_mask = df["Stars Count"] >= 5
    helper.printValueCountsPercentage(stars_mask)

    print("status:")
    helper.printValueCountsPercentage(df["Status"])
    active_mask = np.logical_or(df["Status"].isna(), df["Status"] == "Active")

    print("fork:")
    helper.printValueCountsPercentage(df["Fork"])
    non_fork_mask = df["Fork"] == False

    mask = np.logical_and(np.logical_and(stars_mask, active_mask), non_fork_mask)
    print("total mask:")
    helper.printValueCountsPercentage(mask)
    filtered_df = df[mask]
    print(len(filtered_df))
    filtered_df.to_csv(commonpath.OUTPUT_DIR / "not_toy_ml_repos.csv", index=False)


def main():
    # collect_popular_repos()
    collect_ml_repos()
    filter_out_toy_projects()


if __name__ == '__main__':
    main()
