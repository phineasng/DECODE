import torch
import numpy as np
from Levenshtein import distance as lev_dist
from scipy.spatial.distance import pdist, squareform
from pytoda.proteins import ProteinFeatureLanguage
from pytoda.smiles import SMILESLanguage
from pytoda.datasets import DrugAffinityDataset
from pytoda.proteins.processing import BLOSUM62


AA_LANG_KEY = 'protein_language_fpath'
AFF_KEY = 'affinity_fpath'
PEP_FPATH = 'pep_fpath'
PROTEIN_FPATH = 'protein_fpath'
SMI_LANG_KEY = 'smile_language_fpath'
SETUP_PARAMS_KEY = 'setup_params'

BLOSUM_KEY_MAP = {k: k for k in BLOSUM62.keys()}
BLOSUM_IDX2KEY = [k for k in sorted(BLOSUM62.keys())]
BLOSUM_KEY2IDX = {k: i for i,k in enumerate(BLOSUM_IDX2KEY)}
BLOSUM_KEY_MAP['<PAD>'] = '_'
BLOSUM_KEY_MAP['<UNK>'] = '_' # In pytoda, UNK has the same encoding as pad
BLOSUM_KEY_MAP['<START>'] = '<'
BLOSUM_KEY_MAP['<STOP>'] = '>'
REVERSE_BLOSUM62 = {
    np.array(v).tobytes(): BLOSUM_KEY_MAP[k] for k, v in BLOSUM62.items()
}
REVERSE_BLOSUM62_INT = {
    np.array(v).tobytes(): BLOSUM_KEY2IDX[k] for k, v in BLOSUM62.items()
}


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


def get_aa_tcr_dataset(data_params, device=torch.device('cpu')):
    protein_language_file = data_params[AA_LANG_KEY]
    smiles_language_file = data_params[SMI_LANG_KEY]
    affinity_filepath = data_params[AFF_KEY]
    pep_filepath = data_params[PEP_FPATH]
    protein_filepath = data_params[PROTEIN_FPATH]
    setup_params = data_params[SETUP_PARAMS_KEY]

    # set languages
    protein_language = ProteinFeatureLanguage.load(protein_language_file)
    smiles_language = SMILESLanguage.load(smiles_language_file)
    smiles_language.set_encoding_transforms(
        randomize=None,
        add_start_and_stop=setup_params.get('ligand_start_stop_token', True),
        padding=setup_params.get('ligand_padding', True),
        padding_length=setup_params.get('ligand_padding_length', True),
        device=device,
    )
    smiles_language.set_smiles_transforms(
        augment=setup_params.get('augment_smiles', False),
        canonical=setup_params.get('smiles_canonical', False),
        kekulize=setup_params.get('smiles_kekulize', False),
        all_bonds_explicit=setup_params.get('smiles_bonds_explicit', False),
        all_hs_explicit=setup_params.get('smiles_all_hs_explicit', False),
        remove_bonddir=setup_params.get('smiles_remove_bonddir', False),
        remove_chirality=setup_params.get('smiles_remove_chirality', False),
        selfies=setup_params.get('selfies', False),
        sanitize=setup_params.get('sanitize', False)
    )

    # create dataset
    dataset = DrugAffinityDataset(
        drug_affinity_filepath=affinity_filepath,
        smi_filepath=pep_filepath,
        protein_filepath=protein_filepath,
        smiles_language=smiles_language,
        protein_language=protein_language,
        smiles_padding=setup_params.get('ligand_padding', True),
        smiles_padding_length=setup_params.get('ligand_padding_length', None),
        smiles_add_start_and_stop=setup_params.get(
            'ligand_add_start_stop', True
        ),
        smiles_augment=setup_params.get('augment_smiles', False),
        smiles_canonical=setup_params.get('smiles_canonical', False),
        smiles_kekulize=setup_params.get('smiles_kekulize', False),
        smiles_all_bonds_explicit=setup_params.get(
            'smiles_bonds_explicit', False
        ),
        smiles_all_hs_explicit=setup_params.get('smiles_all_hs_explicit', False),
        smiles_remove_bonddir=setup_params.get('smiles_remove_bonddir', False),
        smiles_remove_chirality=setup_params.get(
            'smiles_remove_chirality', False
        ),
        smiles_selfies=setup_params.get('selfies', False),
        protein_amino_acid_dict=setup_params.get(
            'protein_amino_acid_dict', 'iupac'
        ),
        protein_padding=setup_params.get('receptor_padding', True),
        protein_padding_length=setup_params.get('receptor_padding_length', None),
        protein_add_start_and_stop=setup_params.get(
            'receptor_add_start_stop', True
        ),
        protein_augment_by_revert=setup_params.get('augment_protein', False),
        device=device,
        drug_affinity_dtype=torch.float,
        backend='eager',
        iterate_dataset=False
    )

    dataset_numpy = []
    y_numpy = []
    for i in range(len(dataset)):
        dataset_numpy.append(dataset[i][1].detach().cpu().numpy())
        y_numpy.append(dataset[i][2].detach().cpu().numpy())

    return np.stack(dataset_numpy, axis=0), np.array(y_numpy)


def blosum_embedding2str(x):
    """
    Translates a 2D sample (encoded with BLOSUM62 predefined embeddings) to string
    """
    seq = ''.join([REVERSE_BLOSUM62[t.astype(np.int).tobytes()] for t in x])
    return seq


def _flattenblosumembedding2levenshtein(x, y):
    """
    Translates a 2D sample (encoded with BLOSUM62 predefined embeddings) to string
    """
    return lev_dist(blosum_embedding2str(x.reshape(-1, 26)), blosum_embedding2str(y.reshape(-1, 26)))


def blosum2levenshtein(samples):
    """
    Turn blosum encoded samples to levenshtein matrix

    Args:
        samples (np.array): samples to convert
    """
    flatten_samples = samples.reshape(len(samples), -1)
    return squareform(pdist(flatten_samples, metric=_flattenblosumembedding2levenshtein))


def blosum_embedding2idx(samples):
    """
    Turn blosum encoded samples to categorical idx
    """
    new_x = []
    for sample in samples:
        new_x.append([REVERSE_BLOSUM62_INT[t.astype(np.int).tobytes()] for t in sample])
    return np.stack(new_x, axis=0)
