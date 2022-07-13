import csv
import numpy as np
import pandas as pd
from io import StringIO
import os
from collections import Counter
from keras import backend as K
from Levenshtein import distance as lev_dist
from scipy.spatial.distance import pdist, squareform


########################### One Hot ##########################
aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19,'X': 20}  # 'X' is a padding variable
########################### Blosum ##########################
BLOSUM50_MATRIX = pd.read_table(StringIO(u"""                                                                                      
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *                                                           
A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -2 -1 -1 -5                                                           
R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1 -3  0 -1 -5                                                           
N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  5 -4  0 -1 -5                                                           
D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  6 -4  1 -1 -5                                                           
C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -2 -3 -1 -5                                                           
Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0 -3  4 -1 -5                                                           
E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1 -3  5 -1 -5                                                           
G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -4 -2 -1 -5                                                           
H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0 -3  0 -1 -5                                                          
I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4  4 -3 -1 -5                                                           
L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4  4 -3 -1 -5                                                           
K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0 -3  1 -1 -5                                                           
M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3  2 -1 -1 -5                                                           
F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4  1 -4 -1 -5                                                           
P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -3 -1 -1 -5                                                           
S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0 -3  0 -1 -5                                                           
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1 -1 -1 -5                                                           
W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -2 -1 -5                                                           
Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -1 -2 -1 -5                                                           
V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -3  2 -3 -1 -5                                                           
B -2 -1  5  6 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -3  6 -4  1 -1 -5                                                           
J -2 -3 -4 -4 -2 -3 -3 -4 -3  4  4 -3  2  1 -3 -3 -1 -2 -1  2 -4  4 -3 -1 -5                                                           
Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  1 -3  5 -1 -5                                                           
X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -5                                                           
* -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1                                                           
"""), sep='\s+').loc[list(aa_dict_one_hot.keys()), list(aa_dict_one_hot.keys())]
assert (BLOSUM50_MATRIX == BLOSUM50_MATRIX.T).all().all()

ENCODING_DATA_FRAMES = {
    "BLOSUM50": BLOSUM50_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(aa_dict_one_hot.keys()))]
        for j in range(len(aa_dict_one_hot.keys()))
    ], index=aa_dict_one_hot.keys(), columns=aa_dict_one_hot.keys())
}



DEFAULT_ENCODING_METHOD = 'BLOSUM50'
DEFAULT_ANTIGEN_MAXLEN = 15


def preprocess(filedir, HLA_seq_lib):
    #Preprocess TCR files
    print('Processing: '+filedir)
    if not os.path.exists(filedir):
        print('Invalid file path: ' + filedir)
        return 0
    dataset = pd.read_csv(filedir, header=0)
    dataset = dataset.sort_values('CDR3').reset_index(drop=True)
    #Preprocess HLA_antigen files
    #remove HLA which is not in HLA_seq_lib; if the input hla allele is not in HLA_seq_lib; then the first HLA startswith the input HLA allele will be given
    #Remove antigen that is longer than 15aa
    dataset=dataset.dropna()
    HLA_list=set(dataset['HLA'])
    HLA_to_drop = list()
    for i in HLA_list:
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(i))])==0:
            HLA_to_drop.append(i)
            print('drop '+i)
    dataset=dataset[~dataset['HLA'].isin(HLA_to_drop)]
    dataset=dataset[dataset.Antigen.str.len()<16]
    print(str(max(dataset.index)-dataset.shape[0]+1)+' antigens longer than 15aa are dropped!')
    TCR_list=dataset['CDR3'].tolist()
    antigen_list=dataset['Antigen'].tolist()
    HLA_list=dataset['HLA'].tolist()
    if 'label' in dataset.columns:
        labels = dataset['label']
    return TCR_list,antigen_list,HLA_list,labels


def aamapping_TCR(peptideSeq,aa_dict):
    #Transform aa seqs to Atchley's factors.
    peptideArray = []
    if len(peptideSeq)>80:
        print('Length: '+str(len(peptideSeq))+' over bound!')
        peptideSeq=peptideSeq[0:80]
    for aa_single in peptideSeq:
        try:
            peptideArray.append(aa_dict[aa_single])
        except KeyError:
            print('Not proper aaSeqs: '+peptideSeq)
            peptideArray.append(np.zeros(5,dtype='float32'))
    for i in range(0,80-len(peptideSeq)):
        peptideArray.append(np.zeros(5,dtype='float32'))
    return np.asarray(peptideArray)


def TCRMap(dataset,aa_dict):
    # Wrapper of aamapping
    pos = 0
    TCR_counter = Counter(dataset)
    TCR_array = np.zeros((len(dataset), 80, 5, 1), dtype=np.float32)
    for sequence, length in TCR_counter.items():
        TCR_array[pos:pos+length] = np.repeat(aamapping_TCR(sequence, aa_dict).reshape(1,80,5,1), length, axis=0)
        pos += length
    return TCR_array


def create_hla_seq_lib(hla_db_dir):
    HLA_ABC = [hla_db_dir + '/A_prot.fasta', hla_db_dir + '/B_prot.fasta', hla_db_dir + '/C_prot.fasta',
               hla_db_dir + '/E_prot.fasta']
    HLA_seq_lib = {}
    for one_class in HLA_ABC:
        prot = open(one_class)
        # pseudo_seq from netMHCpan:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796;
        # minor bug 33 aa are used for pseudo seq, the performance is still good
        pseudo_seq_pos = [7, 9, 24, 45, 59, 62, 63, 66, 67, 79, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116,
                          118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]
        # write HLA sequences into a library
        # class I alles
        name = ''
        sequence = ''
        for line in prot:
            if len(name) != 0:
                if line.startswith('>HLA'):
                    pseudo = ''
                    for i in range(0, 33):
                        if len(sequence) > pseudo_seq_pos[i]:
                            pseudo = pseudo + sequence[pseudo_seq_pos[i]]
                    HLA_seq_lib[name] = pseudo
                    name = line.split(' ')[1]
                    sequence = ''
                else:
                    sequence = sequence + line.strip()
            else:
                name = line.split(' ')[1]
    return HLA_seq_lib


def load_aatchley_dict(aatchley_dir, return_idx_maps=False):
    aa_dict_atchley=dict()
    with open(aatchley_dir,'r') as aa:
        aa_reader=csv.reader(aa)
        next(aa_reader, None)
        for rows in aa_reader:
            aa_name=rows[0]
            aa_factor=rows[1:len(rows)]
            aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
    reversed_dictionary = _reverse_embedding_dictionary(aa_dict_atchley)
    if return_idx_maps:
        idx2key = [k for k in aa_dict_atchley.keys()]
        idx2key.append('X')
        key2idx = {k: i for i, k in enumerate(idx2key)}
        reversed_dictionary_int = {k: key2idx[v] for k, v in reversed_dictionary.items()}
        return aa_dict_atchley, reversed_dictionary, idx2key, key2idx, reversed_dictionary_int
    return aa_dict_atchley, reversed_dictionary


def float_array_to_key(a):
    return np.around(a, decimals=2).astype(np.float32).tobytes()


def _reverse_embedding_dictionary(d):
    reversed_dict = {float_array_to_key(v): k for k, v in d.items()}
    for v in d.values():
        zero_key = float_array_to_key(np.zeros(*(v.shape)))
        reversed_dict[zero_key] = 'X'
    return reversed_dict


def aatchley2str(sample, reversed_dict):
    return ''.join([reversed_dict[float_array_to_key(factor)] for factor in sample])


def _flattenaatchleyembedding2levenshtein(x, y, reverse_aatchley_dict):
    return lev_dist(aatchley2str(x.reshape(-1, 5, 1), reverse_aatchley_dict),
                    aatchley2str(y.reshape(-1, 5, 1), reverse_aatchley_dict))


def aatchley2levenshtein(samples, *, model, **kwargs):
    reversed_dict = model._aatchley_dict_reversed
    flatten_samples = samples.reshape(len(samples), -1)
    distance_matrix = squareform(pdist(flatten_samples,
                                       metric=lambda x, y: _flattenaatchleyembedding2levenshtein(x, y, reversed_dict)))
    return distance_matrix


def aatchley_pmtnet_embedding2idx(samples, *, model, **kwargs):
    reverse_aatchley_int = model._aatchley_dict_reversed_int
    new_x = []
    for sample in samples:
        new_x.append([reverse_aatchley_int[float_array_to_key(t)] for t in sample])
    return np.stack(new_x, axis=0)


def hla_encode(HLA_name, HLA_seq_lib, encoding_method):
    #Convert the a HLA allele to a zero-padded numeric representation.
    if HLA_name not in HLA_seq_lib.keys():
        HLA_name=[hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))][0]
    if HLA_name not in HLA_seq_lib.keys():
        print('Not proper HLA allele:'+HLA_name)
    HLA_sequence=HLA_seq_lib[HLA_name]
    HLA_int=[aa_dict_one_hot[char] for char in HLA_sequence]
    if len(HLA_int)!=34:
        k=len(HLA_int)
        HLA_int.extend([20] * (34 - k))
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[HLA_int]
    # Get a numpy array of 34 rows and 21 columns
    return np.asarray(result)


def peptide_encode_HLA(peptide, maxlen, encoding_method):
    #Convert peptide amino acid sequence to numeric encoding
    if len(peptide) > maxlen:
        msg = 'Peptide %s has length %d > maxlen = %d.'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    peptide= peptide.replace(u'\xa0', u'').upper()    #remove non-breaking space
    o = [aa_dict_one_hot[aa] if aa in aa_dict_one_hot.keys() else 20 for aa in peptide]
    #if the amino acid is not valid, replace it with padding aa 'X':20
    k = len(o)
    #use 'X'(20) for padding
    o = o[:k // 2] + [20] * (int(maxlen) - k) + o[k // 2:]
    if len(o) != maxlen:
        msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
        raise ValueError(msg % (peptide, len(peptide), maxlen))
    result=ENCODING_DATA_FRAMES[encoding_method].iloc[o]
    return np.asarray(result)


def preprocess_antigen(dataset, maxlen=DEFAULT_ANTIGEN_MAXLEN, encoding_method=DEFAULT_ENCODING_METHOD):
    #Input a list of antigens and get a three dimentional array
    pos=0
    antigen_array = np.zeros((len(dataset), maxlen, 21), dtype=np.int8)
    antigens_seen = dict()
    for antigen in dataset:
        if antigen not in antigens_seen.keys():
            antigen_array[pos]=peptide_encode_HLA(antigen, maxlen,encoding_method).reshape(1,maxlen,21)
            antigens_seen[antigen] = antigen_array[pos]
        else:
            antigen_array[pos] = antigens_seen[antigen]
        pos += 1
    print('antigenMap done!')
    return antigen_array


def preprocess_hla(dataset, HLA_seq_lib, encoding_method=DEFAULT_ENCODING_METHOD):
    #Input a list of HLA and get a three dimentional array
    pos=0
    HLA_array = np.zeros((len(dataset), 34, 21), dtype=np.int8)
    HLA_seen = dict()
    for HLA in dataset:
        if HLA not in HLA_seen.keys():
            HLA_array[pos] = hla_encode(HLA, HLA_seq_lib, encoding_method).reshape(1,34,21)
            HLA_seen[HLA] = HLA_array[pos]
        else:
            HLA_array[pos] = HLA_seen[HLA]
        pos += 1
    print('HLAMap done!')
    return HLA_array


def get_pmtnet_dataset(fpath, aatchley_dir, hla_lib_dir):
    HLA_seq_lib = create_hla_seq_lib(hla_lib_dir)
    tcr, antigen, hla, labels = preprocess(fpath, HLA_seq_lib)
    aa_dict_atchley, _ = load_aatchley_dict(aatchley_dir)
    TCR_array = TCRMap(tcr, aa_dict_atchley)
    antigen_array = preprocess_antigen(antigen)
    HLA_array = preprocess_hla(hla, HLA_seq_lib)
    return {
        'tcr': TCR_array,
        'antigen': antigen_array,
        'hla': HLA_array
    }, labels


def get_pmtnet_cdr3_only_dataset(dataset_params):
    dataset, labels = get_pmtnet_dataset(**dataset_params)
    return dataset['tcr'], labels


def pearson_correlation_f(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred) #being K.mean a scalar here, it will be automatically subtracted from all elements in y_pred
    fst = y_true - K.mean(y_true)
    devP = K.std(y_pred)
    devT = K.std(y_true)
    return K.mean(fsp*fst)/(devP*devT)
