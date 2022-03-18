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

### Install

- Clone this repository and access it

```bash
git clone --recurse-submodules <LINK2UPDATE> decode
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
The `-e` flag is optional, and suggested only for development.
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
run_interpretability_pipeline --config <PATH_TO_CONFIG_FILE>
```

or, alternatively, from the repo root directory

```bash
python bin/run_interpretability_pipeline --config <PATH_TO_CONFIG_FILE>
```

The `--help` flag can be used to show further options.

## Tutorials/Submission reproducibility

- [TITAN](misc/tutorials/TITAN.md)
- [pMTnet](misc/tutorials/pMTnet.md)

## How to customize

- [Add your own model](tbd)
- [Add your own dataset](tbd)

## Cite

If you are using our work, please cite us.  

```
@{tbd
}
```
