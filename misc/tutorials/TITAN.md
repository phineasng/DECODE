# Running TITAN

We provide a config file and a pretrained model to reproduce 
the experiment on TITAN reported in our paper. 

## Steps to reproduce the experiment

0. Be sure you installed TITAN's dependencies

```bash
pip install extern/titan
pip install extern/pytoda
```

1. Download the trained model at [https://ibm.box.com/shared/static/g5gfmghuwpg8osr4mljk8zp8uvar9o9e.pt](https://ibm.box.com/shared/static/g5gfmghuwpg8osr4mljk8zp8uvar9o9e.pt)
2. To use our config file out-of-the-box, place the file at the path
```bash
<PATH_TO_REPO>/data/titan_model/smi_full_fold1_published_code/weights/best_ROC-AUC_bimodal_mca.pt
```
3. In the provided config file at `<PATH_TO_REPO>/sample_configs/titan.json`, 
   substitute all the occurences of `<PATH_TO_REPO>` to the actual path where you cloned this repository.
   
4. In the provided config file at `<PATH_TO_REPO>/sample_configs/titan.json`, change the `root` entry to a folder where you want to store the results.

5. (optional) If you installed `DECODE` in a conda environment, activate the environment, e.g.
   (if your environment is called `decode`)
   
```bash
conda activate decode
```

6. Run `DECODE`

```bash
run_interpretability_pipeline --config <PATH_TO_REPO>/sample_configs/titan.json --user_directory <PATH_TO_REPO>/example/titan
```

