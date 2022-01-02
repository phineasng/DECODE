from intcr.data.tcr_titan import get_aa_tcr_dataset


DATASET_CONFIG_KEY = 'dataset'
DATASET_KEY = 'dataset_id'
DATASET_PARAMS_KEY = 'dataset_params'
DATASET_GETTERS = {
    'tcr_affinity_smile': get_aa_tcr_dataset,
}
