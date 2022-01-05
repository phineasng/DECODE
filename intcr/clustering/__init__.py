from sklearn.cluster import (KMeans, DBSCAN, MiniBatchKMeans,
                             SpectralClustering, MeanShift, AgglomerativeClustering,
                             Birch, OPTICS, SpectralBiclustering, SpectralCoclustering)
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding
from intcr.data.tcr_titan import blosum2levenshtein, blosum_embedding2idx
from sklearn_extra.cluster import KMedoids


CLUSTERING_KEY = 'clustering'
PRECLUST_TRANSFORM_KEY = 'cluster_preproc_transform'
CLUSTERING_ALGOS_KEY = 'algos'
CONSENSUS_KEY = 'consensus'
SELECTION_KEY = 'cluster_selection'
VISUALIZATION_KEY = 'cluster_visualization'


PRE_CLUSTER_TRANSFORM_REGISTRY = {
    'blosum_emb2levenshtein': blosum2levenshtein,
    'blosum_emb2categorical': blosum_embedding2idx
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


CLUSTERING_ALGOS_CENTERS_GET_FN = {
    'KMeans': lambda model: None,
    'KMedoids': lambda model: model.medoid_indices_,
    'DBSCAN': lambda model: model.core_sample_indices_,
    'MiniBatchKMeans': lambda model: None,
    'SpectralClustering': lambda model: None,
    'MeanShift': lambda model: None,
    'AgglomerativeClustering': lambda model: None,
    'Birch': lambda model: None,
    'OPTICS': lambda model: None,
    'SpectralBiclustering': lambda model: None,
    'SpectralCoclustering': lambda model: None
}


CLUSTERING_EVALUATION_REGISTRY = {
    'davies_bouldin': davies_bouldin_score,
    'calinski_harabasz': calinski_harabasz_score,
    'silhouette': silhouette_score
}

CLUSTERING_EVALUATION_MULTIPLIER = {
    'davies_bouldin': -1,
    'calinski_harabasz': 1,
    'silhouette': 1
}


DATA_VISUALIZATION_REGISTRY = {
    'tsne': TSNE,
    'spectral': SpectralEmbedding,
    'LLE': LocallyLinearEmbedding
}
