from intcr.data.tcr_titan import get_aa_tcr_dataset, BLOSUM_IDX2KEY


DATASET_CONFIG_KEY = 'dataset'
DATASET_KEY = 'dataset_id'
DATASET_PARAMS_KEY = 'dataset_params'
DATASET_GETTERS = {
    'tcr_affinity_smile': get_aa_tcr_dataset,
}
CATEGORICAL_ALPHABETS = {
    'blosum_categorical': BLOSUM_IDX2KEY
}
