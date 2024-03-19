#!/usr/bin/env python

import pathlib

import commonpath
import helper

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


def extractVersions(projectIDs, savePath: pathlib.Path):
    if savePath.exists():
        raise helper.FileAlreadyExist(f"{savePath} already exist!")
    first = True
    for idx, chunk in enumerate(helper.readCSV(
            commonpath.LIBRARIES_IO_VERSIONS_PATH,
            chunksize=100000,
    )):
        print(idx)
        masks = chunk["Project ID"].isin(projectIDs)
        versions = chunk[masks]
        if len(versions):
            versions.to_csv(savePath, mode='a', header=first, index=False)
            first = False


def extractTags(repoIDs, savePath: pathlib.Path):
    if savePath.exists():
        raise helper.FileAlreadyExist(f"{savePath} already exist!")
    first = True
    for idx, chunk in enumerate(helper.readCSV(
            commonpath.LIBRARIES_IO_TAGS_PATH,
            chunksize=100000,
    )):
        print(idx)
        masks = chunk["Repository ID"].isin(repoIDs)
        versions = chunk[masks]
        if len(versions):
            versions.to_csv(savePath, mode='a', header=first, index=False)
            first = False


def extractBindingVersions():
    # binding_project_ids = helper.read_json(commonpath.OUTPUT_DIR / "binding_project_ids.json")
    # extractVersions(binding_project_ids, commonpath.OUTPUT_DIR / "binding_project_versions.csv")
    binding_project_df = helper.readCSV(commonpath.OUTPUT_DIR / "ml_cross_ecosystem_packages.csv")
    extractVersions(binding_project_df["ID"], commonpath.OUTPUT_DIR / "ml_cross_ecosystem_packages_versions.csv")


def main():
    extractBindingVersions()


if __name__ == '__main__':
    main()
