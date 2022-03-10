from intcr.extern.pmtnet.pMTnet import preprocess, aa_dict_atchley, TCRMap, antigenMap, HLAMap


def preprocess_antigen(antigen_list):
    return antigenMap(antigen_list, 15, 'BLOSUM50')


def preprocess_hla(hla_list):
    return HLAMap(hla_list, 'BLOSUM50')


def get_pmtnet_dataset(fpath):
    tcr, antigen, hla = preprocess(fpath)
    TCR_array = TCRMap(tcr, aa_dict_atchley)
    antigen_array = preprocess_antigen(antigen)
    HLA_array = preprocess_hla(hla)
    return {
        'tcr': TCR_array,
        'antigen': antigen_array,
        'hla': HLA_array
    }, None


def get_pmtnet_cdr3_only_dataset(fpath):
    return get_pmtnet_dataset(fpath)[0]['tcr']
