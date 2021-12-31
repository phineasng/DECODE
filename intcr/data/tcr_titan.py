import torch
import inspect
import numpy as np
from pytoda.proteins import ProteinFeatureLanguage
from pytoda.smiles import SMILESLanguage
from pytoda.datasets import DrugAffinityDataset


AA_LANG_KEY = 'protein_language_fpath'
AFF_KEY = 'affinity_fpath'
PEP_FPATH = 'pep_fpath'
PROTEIN_FPATH = 'protein_fpath'
SMI_LANG_KEY = 'smile_language_fpath'
SETUP_PARAMS_KEY = 'setup_params'


def _map_dict_keys(setup_params):
    """
    Mapping keys utility for compatibility with TITAN code
    """
    def _map_key(d, old_key, new_key):
        d[new_key] = d.pop(old_key)
        return d

    setup_params = _map_key(setup_params, 'smiles_start_stop_token', 'add_start_and_stop')
    setup_params = _map_key(setup_params, 'smiles_bonds_explicit', 'smiles_all_bonds_explicit')
    setup_params = _map_key(setup_params, 'selfies', 'smiles_selfies')
    setup_params = _map_key(setup_params, 'smiles_add_start_stop', 'smiles_add_start_and_stop')
    setup_params = _map_key(setup_params, 'protein_add_start_stop', 'protein_add_start_and_stop')
    return setup_params


def _filldefault_and_rename_params(setup_params):
    """
    Preprocessing utils to make code cleaner, while maintaining compatibility with TITAN's released code
    """
    default_params = {
        # smile language
        'randomize': None,
        'smiles_start_stop_token': True,

        # drug affinity dataset
        'smiles_add_start_stop': True,
        'smiles_padding': True,
        'smiles_padding_length': 500,
        'smiles_augment': False,
        'smiles_canonical': False,
        'smiles_kekulize': False,
        'smiles_bonds_explicit': False,
        'smiles_all_hs_explicit': False,
        'smiles_remove_bonddir': False,
        'smiles_remove_chirality': False,
        'selfies': False,
        'protein_amino_acid_dict': 'iupac',
        'protein_padding': True,
        'protein_padding_length': None,
        'protein_add_start_stop': True,
        'protein_augment_by_revert': False,
        'drug_affinity_dtype': torch.float,
        'backend': 'eager'

    }
    default_params.update(setup_params)
    default_params = _map_dict_keys(default_params)

    default_params['padding'] = default_params['smiles_padding']
    default_params['padding_length'] = default_params['smiles_padding_length']

    return default_params


def _get_parameters(fn, params):
    args = inspect.getfullargspec(fn)[0]
    fn_params = {k: params[k] for k in args if k in params}
    return fn_params


def get_aa_tcr_dataset(data_params, device=torch.device('cpu')):
    protein_language_file = data_params[AA_LANG_KEY]
    smiles_language_file = data_params[SMI_LANG_KEY]
    affinity_filepath = data_params[AFF_KEY]
    pep_filepath = data_params[PEP_FPATH]
    protein_filepath = data_params[PROTEIN_FPATH]
    setup_params = data_params[SETUP_PARAMS_KEY]
    setup_params = _filldefault_and_rename_params(setup_params)

    # set languages
    protein_language = ProteinFeatureLanguage.load(protein_language_file)
    smiles_language = SMILESLanguage.load(smiles_language_file)
    smiles_language.set_encoding_transforms(
        **(_get_parameters(smiles_language.set_encoding_transforms, setup_params)),
        device=device
    )

    # create dataset
    dataset = DrugAffinityDataset(
        drug_affinity_filepath=affinity_filepath,
        smi_filepath=pep_filepath,
        protein_filepath=protein_filepath,
        smiles_language=smiles_language,
        protein_language=protein_language,
        **(_get_parameters(DrugAffinityDataset, setup_params)),
        device=device
    )

    return dataset


def blosum2index_drugaffinity_ds(samples, model, dataset):
    """
    Turn blosum encoded samples to indeces of an alphabet provided through a DrugAffinity dataset

    Args:
        samples (np.array): samples to convert
        model: unused. Kept to follow a common interface of transforms
        dataset (DrugAffinityDataset): dataset with info to convert from blosum to protein indeces
    """
    language = dataset.protein_sequence_dataset.protein_language
    for s in samples:


