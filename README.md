# Quality estimation

This repository contains scripts to fit feature-based quality estimation model. For this to work in a reproducible way,
you first need to create a conda environment:

```
$ conda env create -f environment.yml
```
Once conda env is created, activate it by running `conda activate qe`. After that run `pip install .` 
to install the package.

Once the installation is completed, you can run `scripts/run.sh` to produce the models.

## Models

The quality models available are:

- [English-Czech](model/csen/encs.quality.lr/)
- [English-Estonian](model/eten/enet.quality.lr/)
- [English-Spanish](model/esen/enes.quality.lr/)

## Tools

- [Convert json quality model file to a binary file and vice versa](tools/lr/)
