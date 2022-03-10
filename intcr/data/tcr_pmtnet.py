from intcr.extern.pmtnet.pMTnet import preprocess, TCRMap, aa_dict_one_hot, ENCODING_DATA_FRAMES
import csv
import numpy as np


DEFAULT_ENCODING_METHOD = 'BLOSUM50'
DEFAULT_ANTIGEN_MAXLEN = 15


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


def load_aatchley_dict(aatchley_dir):
    aa_dict_atchley=dict()
    with open(aatchley_dir,'r') as aa:
        aa_reader=csv.reader(aa)
        next(aa_reader, None)
        for rows in aa_reader:
            aa_name=rows[0]
            aa_factor=rows[1:len(rows)]
            aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')
    return aa_dict_atchley


def hla_encode(HLA_name, hla_lib_dir, encoding_method):
    HLA_seq_lib = create_hla_seq_lib(hla_lib_dir)
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


def preprocess_hla(dataset, hla_lib_dir, encoding_method=DEFAULT_ENCODING_METHOD):
    #Input a list of HLA and get a three dimentional array
    pos=0
    HLA_array = np.zeros((len(dataset), 34, 21), dtype=np.int8)
    HLA_seen = dict()
    for HLA in dataset:
        if HLA not in HLA_seen.keys():
            HLA_array[pos] = hla_encode(HLA, hla_lib_dir, encoding_method).reshape(1,34,21)
            HLA_seen[HLA] = HLA_array[pos]
        else:
            HLA_array[pos] = HLA_seen[HLA]
        pos += 1
    print('HLAMap done!')
    return HLA_array


def get_pmtnet_dataset(fpath, aatchley_dir, hla_lib_dir):
    tcr, antigen, hla = preprocess(fpath)
    aa_dict_atchley = load_aatchley_dict(aatchley_dir)
    TCR_array = TCRMap(tcr, aa_dict_atchley)
    antigen_array = preprocess_antigen(antigen)
    HLA_array = preprocess_hla(hla, hla_lib_dir)
    return {
        'tcr': TCR_array,
        'antigen': antigen_array,
        'hla': HLA_array
    }, None


def get_pmtnet_cdr3_only_dataset(fpath, aatchley_dir, hla_lib_dir):
    return get_pmtnet_dataset(fpath, aatchley_dir, hla_lib_dir)[0]['tcr'], None
