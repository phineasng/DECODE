from intcr.data.tcr_pmtnet import get_pmtnet_cdr3_only_dataset


DATASET_CONFIG_KEY = 'dataset'
DATASET_KEY = 'dataset_id'
DATASET_PARAMS_KEY = 'dataset_params'

DATASET_GETTERS = {
    'cdr3_only_pmtnet': get_pmtnet_cdr3_only_dataset,
}

CATEGORICAL_ALPHABETS = {
    'aatchley_pmtnet_categorical': lambda model: model._aatchley_idx2key
}
