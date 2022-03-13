from intcr.data.tcr_titan import get_aa_tcr_dataset, BLOSUM_IDX2KEY
from intcr.data.tcr_pmtnet import get_pmtnet_cdr3_only_dataset


DATASET_CONFIG_KEY = 'dataset'
DATASET_KEY = 'dataset_id'
DATASET_PARAMS_KEY = 'dataset_params'
DATASET_GETTERS = {
    'tcr_affinity_smile': get_aa_tcr_dataset,
    'cdr3_only_pmtnet': get_pmtnet_cdr3_only_dataset,
}
CATEGORICAL_ALPHABETS = {
    'blosum_categorical': BLOSUM_IDX2KEY,
    'aatchley_pmtnet_categorical': lambda model: model._aatchley_idx2key
}
