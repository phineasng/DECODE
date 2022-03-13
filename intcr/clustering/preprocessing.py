"""
Routines for clustering preprocessing
"""
import os
from intcr.pipeline.config import simple_key_check
from intcr.pipeline.utils import save_data
from intcr.clustering import PRE_CLUSTER_TRANSFORM_REGISTRY
from intcr.pipeline.utils import make_split_name, generate_preprocessing_instance


PRE_CLUSTER_TRANSFORM_FN_KEY = 'transform_fn'
PRE_CLUSTER_TRANSFORM_OUTPUT_ID = 'output_type'


def check_pre_cluster_transform_config(config):
    simple_key_check(config, PRE_CLUSTER_TRANSFORM_FN_KEY)
    simple_key_check(config, PRE_CLUSTER_TRANSFORM_OUTPUT_ID)


def pre_clustering_transform(precluster_root, config, split_samples, model, dataset, recompute=False):
    """
    Routine to go through all the transforms to be used for the next clustering step
    """
    if isinstance(config, dict):
        config = [config]

    def parallelizable_fn(root, cfg, split, samples, recomp_flag):
        check_pre_cluster_transform_config(cfg)

        transform_fn_name = cfg[PRE_CLUSTER_TRANSFORM_FN_KEY]
        output_type = cfg[PRE_CLUSTER_TRANSFORM_OUTPUT_ID]

        out_fpath = os.path.join(root, make_split_name(output_type, split))

        if recomp_flag or not os.path.exists(out_fpath):
            transform_fn = PRE_CLUSTER_TRANSFORM_REGISTRY[transform_fn_name]
            transformed_samples = transform_fn(samples[split], model=model)
            save_data(out_fpath, transformed_samples)

    # to rewrite for parallelization
    for c, s in generate_preprocessing_instance(config, split_samples):
        parallelizable_fn(precluster_root, c, s, split_samples, recompute)
