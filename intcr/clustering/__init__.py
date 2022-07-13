from sklearn.cluster import (KMeans, DBSCAN, MiniBatchKMeans,
                             SpectralClustering, MeanShift, AgglomerativeClustering,
                             Birch, OPTICS, SpectralBiclustering, SpectralCoclustering)
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding
from example.titan.data import blosum2levenshtein, blosum_embedding2idx, blosum2nwalign
from intcr.data.tcr_pmtnet import aatchley2levenshtein, aatchley_pmtnet_embedding2idx
from sklearn_extra.cluster import KMedoids
from collections import defaultdict


CLUSTERING_KEY = 'clustering'
PRECLUST_TRANSFORM_KEY = 'cluster_preproc_transform'
CLUSTERING_ALGOS_KEY = 'algos'
CONSENSUS_KEY = 'consensus'
SELECTION_KEY = 'cluster_selection'
VISUALIZATION_KEY = 'cluster_visualization'


PRE_CLUSTER_TRANSFORM_REGISTRY = {
    'blosum_emb2levenshtein': blosum2levenshtein,
    'blosum_emb2nwalign': blosum2nwalign,
    'blosum_emb2categorical': blosum_embedding2idx,
    'aatchley_pmtnet_emb2levenshtein': aatchley2levenshtein,
    'aatchley_pmtnet_emb2categorical': aatchley_pmtnet_embedding2idx
}

CLUSTERING_ALGOS_REGISTRY = {
    'KMeans': KMeans,
    'KMedoids': KMedoids,
    'DBSCAN': DBSCAN,
    'MiniBatchKMeans': MiniBatchKMeans,
    'SpectralClustering': SpectralClustering,
    'MeanShift': MeanShift,
    'AgglomerativeClustering': AgglomerativeClustering,
    'Birch': Birch,
    'OPTICS': OPTICS,
    'SpectralBiclustering': SpectralBiclustering,
    'SpectralCoclustering': SpectralCoclustering
}


CLUSTERING_ALGOS_CENTERS_GET_FN = defaultdict(lambda: lambda model: None) # by default assume model does not assign centroids
CLUSTERING_ALGOS_CENTERS_GET_FN.update({
    'KMedoids': lambda model: model.medoid_indices_,
    'DBSCAN': lambda model: model.core_sample_indices_,
})


CLUSTERING_EVALUATION_REGISTRY = {
    'davies_bouldin': davies_bouldin_score,
    'calinski_harabasz': calinski_harabasz_score,
    'silhouette': silhouette_score
}

CLUSTERING_EVALUATION_MULTIPLIER = defaultdict(lambda: 1) # by default maximize score, during selection
CLUSTERING_EVALUATION_MULTIPLIER.update({
    'davies_bouldin': -1,
    'calinski_harabasz': 1,
    'silhouette': 1
})


DATA_VISUALIZATION_REGISTRY = {
    'tsne': TSNE,
    'spectral': SpectralEmbedding,
    'LLE': LocallyLinearEmbedding
}
