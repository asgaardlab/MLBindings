#!/usr/bin/env python

import pandas
import pathlib

import helper
import commonpath

LIBRARIES_IO_DIR = pathlib.Path("/media/leo/LeoHarddisk/Dataset/libraries_io")

DTYPES_VERSIONS = {
    "ID": int,
    "Platform": str,
    "Project Name": str,
    "Project ID": int,
    "Number": str,
    "Published Timestamp": str,
    "Created Timestamp": str,
    "Updated Timestamp": str,
}


def extractCreatedDates():
    versions_info = pandas.read_csv(
        "/media/leo/LeoHarddisk/Dataset/libraries_io/versions-1.6.0-2020-01-12.csv",
        usecols=["Project ID", "Published Timestamp"], dtype=DTYPES_VERSIONS
    )

    versions_info["Published Timestamp"] = helper.toDatetime(versions_info["Published Timestamp"])
    versions_info = versions_info.sort_values("Published Timestamp")
    proj_created_date = versions_info.groupby("Project ID").apply(lambda x: x["Published Timestamp"].values[0])

    proj_created_date = proj_created_date.reset_index()
    proj_created_date.rename({"Project ID": "ID", 0: "Created Date"}, axis="columns", inplace=True)
    proj_created_date.to_csv(commonpath.OUTPUT_DIR / "proj_created_dates.csv", index=False)


def main():
    extractCreatedDates()


if __name__ == '__main__':
    main()
