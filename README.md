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
pip install .
```

- (optional) From the folder of the repository, further install dependencies for TITAN.
If the user does not want to install these dependencies, titan files should be removed from the code.
  
```bash
pip install extern/titan
pip install extern/pytoda
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

NOTE: if you customized the pipeline, e.g. by adding your own model, you should make your customization available by adding the flag ``--user_directory``, i.e.

```bash
conda activate decode # if you installed the package in a conda environment
run_interpretability_pipeline --config <PATH_TO_CONFIG_FILE> --user_directory YOUR_FOLDER/
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
@article{decode22,
    author = {Papadopoulou, Iliana and Nguyen, An-Phi and Weber, Anna and Martínez, María Rodríguez},
    title = "{DECODE: a computational pipeline to discover T cell receptor binding rules}",
    journal = {Bioinformatics},
    volume = {38},
    number = {Supplement_1},
    pages = {i246-i254},
    year = {2022},
    month = {06},
    abstract = "{Understanding the mechanisms underlying T cell receptor (TCR) binding is of fundamental importance to understanding adaptive immune responses. A better understanding of the biochemical rules governing TCR binding can be used, e.g. to guide the design of more powerful and safer T cell-based therapies. Advances in repertoire sequencing technologies have made available millions of TCR sequences. Data abundance has, in turn, fueled the development of many computational models to predict the binding properties of TCRs from their sequences. Unfortunately, while many of these works have made great strides toward predicting TCR specificity using machine learning, the black-box nature of these models has resulted in a limited understanding of the rules that govern the binding of a TCR and an epitope.We present an easy-to-use and customizable computational pipeline, DECODE, to extract the binding rules from any black-box model designed to predict the TCR-epitope binding. DECODE offers a range of analytical and visualization tools to guide the user in the extraction of such rules. We demonstrate our pipeline on a recently published TCR-binding prediction model, TITAN, and show how to use the provided metrics to assess the quality of the computed rules. In conclusion, DECODE can lead to a better understanding of the sequence motifs that underlie TCR binding. Our pipeline can facilitate the investigation of current immunotherapeutic challenges, such as cross-reactive events due to off-target TCR binding.Code is available publicly at https://github.com/phineasng/DECODE.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac257},
    url = {https://doi.org/10.1093/bioinformatics/btac257},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/Supplement\_1/i246/44269455/btac257.pdf},
}
```
