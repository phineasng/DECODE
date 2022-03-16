from keras.layers import Input,Dense,concatenate,Dropout
from keras.models import Model,load_model
from intcr.data.tcr_pmtnet import pearson_correlation_f
import pandas as pd
import numpy as np
from intcr.data.tcr_pmtnet import preprocess_hla, preprocess_antigen, create_hla_seq_lib, \
    load_aatchley_dict, TCRMap


class PMTnet:
    """
    pMTnet as per reference implementation at https://github.com/tianshilu/pMTnet/blob/master/pMTnet.py (line 279)
    """
    def __init__(self, *, tcr_encoder_fpath, hla_antigen_encoder_fpath, bg_1k_fpath, bg_10k_fpath,
                 classifier_fpath=None, aatchley_dir: str=None, threshold=0.5):
        self._create_classifier(classifier_fpath)
        self._create_tcr_encoder(tcr_encoder_fpath)
        self._create_hla_antigen_encoder(hla_antigen_encoder_fpath)
        self._load_bg_negatives(bg_1k_fpath, bg_10k_fpath)
        self._thr = threshold
        self._aatchley_dict = None
        self._aatchley_idx2key = None
        self._aatchley_key2idx = None
        self._aatchley_dict_reversed = None
        self._aatchley_dict_reversed_int = None
        if aatchley_dir is not None:
            self._aatchley_dict, self._aatchley_dict_reversed, \
            self._aatchley_idx2key, self._aatchley_key2idx,\
                self._aatchley_dict_reversed_int = \
                load_aatchley_dict(aatchley_dir, return_idx_maps=True)

    def _create_classifier(self, classifier_fpath=None):
        hla_antigen_in = Input(shape=(60,), name='hla_antigen_in')
        pos_in = Input(shape=(30,), name='pos_in')
        ternary_layer1_pos = concatenate([pos_in, hla_antigen_in])
        ternary_dense1 = Dense(300, activation='relu')(ternary_layer1_pos)
        ternary_do1 = Dropout(0.2)(ternary_dense1)
        ternary_dense2 = Dense(200, activation='relu')(ternary_do1)
        ternary_dense3 = Dense(100, activation='relu')(ternary_dense2)
        ternary_output = Dense(1, activation='linear')(ternary_dense3)
        self._classifier = Model(inputs=[pos_in, hla_antigen_in], outputs=ternary_output)
        if classifier_fpath is None:
            self._classifier.load_weights(classifier_fpath)

    def _create_tcr_encoder(self, tcr_encoder_fpath):
        TCR_encoder = load_model(tcr_encoder_fpath)
        self._tcr_encoder = Model(TCR_encoder.input,TCR_encoder.layers[-12].output)

    def encode_tcr(self, tcr):
        return self._tcr_encoder.predict(tcr)

    def _create_hla_antigen_encoder(self, hla_antigen_encoder_fpath):
        HLA_antigen_encoder = load_model(hla_antigen_encoder_fpath,
                                         custom_objects={'pearson_correlation_f': pearson_correlation_f})
        self._hla_antigen_encoder = Model(HLA_antigen_encoder.input, HLA_antigen_encoder.layers[-2].output)

    def encode_hla_antigen(self, antigen, hla):
        return self._hla_antigen_encoder.predict([antigen, hla])

    def _load_bg_negatives(self, bg_1k_fpath, bg_10k_fpath):
        self._tcr_neg_df_1k = pd.read_csv(bg_1k_fpath, names=pd.RangeIndex(0, 30, 1), header=None, skiprows=1).to_numpy()
        self._tcr_neg_df_10k = pd.read_csv(bg_10k_fpath, names=pd.RangeIndex(0, 30, 1), header=None, skiprows=1).to_numpy()

    def predict(self, tcr, antigen, hla):
        encoded_tcr = self.encode_tcr(tcr)
        encoded_hla_antigen = self.encode_hla_antigen(antigen, hla)
        return self.predict_with_bg(encoded_tcr, encoded_hla_antigen)

    def predict_with_bg(self, encoded_tcrs, encoded_hlas_antigens):
        ranks = []
        for tcr, hla_antigen in zip(encoded_tcrs, encoded_hlas_antigens):
            ranks.append(self.predict_with_bg_single(tcr, hla_antigen))
        predictions = self._threshold(np.squeeze(np.array(ranks)))
        return predictions

    def _threshold(self, ranks):
        return (ranks < self._thr).astype(np.int)

    def predict_with_bg_single(self, encoded_tcr, encoded_hla_antigen):
        rank, prediction = self._predict_with_bg_single(encoded_tcr, encoded_hla_antigen, self._tcr_neg_df_1k)
        if rank < 0.02:
            rank, prediction = self._predict_with_bg_single(encoded_tcr, encoded_hla_antigen, self._tcr_neg_df_10k)
        return rank

    def _rank(self, predictions):
        return 1 - np.float(sorted(predictions).index(predictions[0])+1)/len(predictions)

    def _predict_with_bg_single(self, encoded_tcr, encoded_hla_antigen, bg):
        tcr_in = np.concatenate([np.expand_dims(encoded_tcr, axis=0), bg], axis=0)
        hla_antigen_in = np.repeat(np.expand_dims(encoded_hla_antigen, axis=0), len(bg) + 1, axis=0)
        prediction = self._classifier.predict({'pos_in': tcr_in, 'hla_antigen_in': hla_antigen_in})
        rank = self._rank(prediction)
        return rank, prediction


class PMTnetFixedHLAAntigen(PMTnet):
    def __init__(self, *, antigen: str, hla: str, hla_lib_dir: str, **kwargs):
        super(PMTnetFixedHLAAntigen, self).__init__(**kwargs)
        self._preprocess_hla_antigen(hla, antigen, hla_lib_dir)
        self._preprocess_background()

    def _preprocess_hla_antigen(self, hla, antigen, hla_lib_dir):
        self._hla_str = hla
        self._antigen_str = antigen
        self._hla_library = create_hla_seq_lib(hla_lib_dir)
        self._hla_mapped = preprocess_hla([self._hla_str], HLA_seq_lib=self._hla_library)
        self._antigen_mapped = preprocess_antigen([self._antigen_str])
        self._encoded_hla_antigen = self.encode_hla_antigen(self._antigen_mapped, self._hla_mapped)

    def _preprocess_background(self):
        bg_hla = np.repeat(self._encoded_hla_antigen, len(self._tcr_neg_df_1k), axis=0)
        self._bg_1k_preds = np.squeeze(self._classifier.predict({'pos_in': self._tcr_neg_df_1k, 'hla_antigen_in': bg_hla}))
        bg_hla = np.repeat(self._encoded_hla_antigen, len(self._tcr_neg_df_10k), axis=0)
        self._bg_10k_preds = np.squeeze(self._classifier.predict({'pos_in': self._tcr_neg_df_10k, 'hla_antigen_in': bg_hla}))

    def predict(self, tcr, *args):
        if len(tcr.shape) > 2:
            encoded_tcr = self.encode_tcr(tcr)
        else: # in this case, we have an anchor-friendly input, i.e. each sample is an array of indeces
            new_x = []
            for s in tcr:
                sample = []
                for token in s:
                    c = self._aatchley_idx2key[np.int(token)]
                    if c != 'X':
                        sample.append(c)
                new_x.append(''.join(sample))
            new_x = TCRMap(new_x, self._aatchley_dict)
            encoded_tcr = self.encode_tcr(new_x)
        repeated_hla = np.repeat(self._encoded_hla_antigen, len(encoded_tcr), axis=0)
        preds = self._classifier.predict({'pos_in': encoded_tcr, 'hla_antigen_in': repeated_hla})
        preds = np.squeeze(preds, axis=tuple(list(range(len(preds.shape)))[1:]))
        ranks = []
        for i in range(len(preds)):
            curr_pred = preds[i:i+1]
            if len(tcr) == 2:
                print(curr_pred)
            curr_2_rank = np.concatenate([curr_pred, self._bg_1k_preds])
            rank = self._rank(curr_2_rank)
            if rank < 0.02:
                rank = self._rank(np.concatenate([curr_pred, self._bg_10k_preds]))
            ranks.append(rank)
        predictions = self._threshold(np.array(ranks))
        return predictions


def load_pmt_net_fixed_hla_antigen(model_config, *args, **kwargs):
    return PMTnetFixedHLAAntigen(**model_config)
