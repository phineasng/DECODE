from intcr.models.titan import load_titan_fixed_epitope
from intcr.models.pmtnet import load_pmt_net_fixed_hla_antigen


MODEL_CONFIG_KEY = 'model'
MODEL_NAME_KEY = 'model_id'
MODEL_PARAMS_KEY = 'model_params'
MODEL_LOADERS = {
    'titan_fixed_epitope': load_titan_fixed_epitope,
    'pmtnet_fixed_hla_antigen': load_pmt_net_fixed_hla_antigen
}
