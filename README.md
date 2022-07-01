# DECODE: a computational pipeline to discover T-cell receptor binding rules

DECODE is a python package that allows the user to generate 
explanations for a TCR binding model. 

**Features**:

- Works on any black-box model
- Global explanaibility via clustering
- User-friendliness via text-based configuration
- Customizable by [adding your own model/data](#how-to-customize)

## Getting Started

We have tested these instructions in Linux. 
If you are having issues on other platforms, please do not hesitate to contact us.

### Requirements

- python 3.7
- pip

### Install

- Clone this repository and access it

```bash
git clone --recurse-submodules https://github.com/phineasng/DECODE.git decode
cd decode
```

- (optional, but recommended) Create a conda environment 
  (here we call it decode, but any name can be used), and activate it
```bash
conda create --name decode python=3.7
conda activate decode
```


- Manually install some dependencies, according to your preferred settings (e.g. `cpu-only`, `gpu`, ...). 
  
    * `pytorch` ([website](https://pytorch.org/))
      - We run our experiments with `torch==1.10`, but some previous versions should work as well (e.g. `1.8`, `1.9`)
      - We suggest to use pip
    

- From the folder of the repository, install `DECODE` and other dependencies.
```bash
pip install -e .
```

- From the folder of the repository, further install dependencies for TITAN.
If the user does not want to install these dependencies, titan files should be removed from the code.
  
```bash
pip install intcr/extern/titan
pip install intcr/extern/pytoda
```

- (optional) From the folder of the repository, the user can also install additional dependencies to help visualization (i.e. alignment).
  
```bash
pip install intcr/extern/anarci
```

## Basic usage

```bash
conda activate decode # if you installed the package in a conda environment
run_interpretability_pipeline --config <PATH_TO_CONFIG_FILE>
```

or, alternatively, from the repo root directory

```bash
conda activate decode # if you installed the package in a conda environment
python bin/run_interpretability_pipeline --config <PATH_TO_CONFIG_FILE>
```

The `--help` flag can be used to show further options.

## Overview of the pipeline

- [Config file]()
- [Data preparation outputs]()
- [Clustering outputs]()
- [Anchors output]()

## Tutorials/Submission reproducibility

- [TITAN](misc/tutorials/TITAN.md)
- [pMTnet](misc/tutorials/pMTnet.md)

## How to customize

- [Add your own dataset](misc/further_instructions/add_dataset.md)
    - [Add your own data preprocessing function](misc/further_instructions/add_dataset.md#processing-the-data) 
    - [Define an alphabet for Anchors](misc/further_instructions/add_dataset.md#anchors-alphabet)
- [Add your own model](misc/further_instructions/add_model.md)
- [Add a clustering algorithm](misc/further_instructions/add_clustering_method.md)
    - [Add a clustering scoring method](misc/further_instructions/add_clustering_method.md#add-a-clustering-scoring-method)

NOTE: Our pipeline design should cover most of the use cases, so it should be relatively simple to add your own models/data. 
If your use case scenario is not covered (e.g. your input samples have to be a dictionary rather than a simple array), please add an issue to discuss a potential change in the pipeline.

## Cite

If you are using our work, please cite us.  

```
@{tbd
}
```
