from intcr.data.tcr_titan import get_aa_tcr_dataset


DATASET_KEY = 'dataset'
DATASET_PARAMS_KEY = 'dataset_params'
DATASET_GETTERS = {
    'tcr_affinity_smile': get_aa_tcr_dataset,
}
