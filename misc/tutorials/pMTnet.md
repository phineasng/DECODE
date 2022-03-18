# Running pMTnet

We provide a config file and a pretrained model to reproduce 
the experiment on pMTnet reported in our paper. 

## Steps to reproduce the experiment

1. In the provided config file at `<PATH_TO_REPO>/sample_configs/pmtnet.json`, 
   substitute all the occurences of `<PATH_TO_REPO>` to the actual path where you cloned this repository.
   
2. In the provided config file at `<PATH_TO_REPO>/sample_configs/pmtnet.json`, change the `root` entry to a folder where you want to store the results.

3. (optional) If you installed `DECODE` in a conda environment, activate the environment, e.g.
   (if your environment is called `decode`)
   
```bash
conda activate decode
```

4. Run `DECODE`

```bash
run_interpretability_pipeline --config <PATH_TO_REPO>/sample_configs/pmtnet.json
```

