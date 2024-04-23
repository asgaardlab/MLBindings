# Replication package for the paper "Bridging the Language Gap: An Empirical Study of Bindings for Open Source Machine Learning Libraries in Software Package Ecosystems	"

## Data Overview

The data is based on the dataset of [Librario.io](https://doi.org/10.5281/zenodo.3626071), the extracted data can be
found in the [data/](./data/) folder:

- [all_bindings.csv](data/all_bindings.csv): Contains 250,668 bindings identified by BindFind, alongside their respective host library names.
- [data/labelled_data](data/labelled_data): This directory contains labelled binding data which split into training, validation, and testing sets.
- [data/binding_qa](data/binding_qa): This directory contains the performance results of BERT-like models on our labelled dataset
- [rq2_ml_repos.csv](data/rq2_ml_repos.csv) and [rq2_ml_bindings.csv](data/rq2_ml_bindings.csv): Provide details on 546 ML libraries and their 2,436 bindings.
- [labelled_rq3_pop_ml_repos.csv](data/labelled_rq3_pop_ml_repos.csv) and [labelled_rq3_pop_ml_bindings.csv](data/labelled_rq3_pop_ml_bindings.csv): Provide details on 40 popular ML libraries and their 133 bindings
- [rq3_pop_ml_repos_tags.csv](data/rq3_pop_ml_repos_tags.csv) and [labelled_rq3_pop_ml_bindings_versions.csv](data/labelled_rq3_pop_ml_bindings_versions.csv): Provide the results of our version matching analysis for 3,785 tags and 3,277 versions.

## Environment Setup

We provide an [environment.yml](environment.yml) file that can be used with Conda to create an environment with all the necessary dependencies:

```
conda env create -f environment.yml
```

## Reproducing the Study

For replicating the results presented in our paper, we have organized Jupyter notebooks in the 
[analyze_notebooks](src/analyze_notebooks) directory. In addition, we provide the scripts for 
data collecting in the [data_collection](src/data_collection) directory


