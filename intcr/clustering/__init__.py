from sklearn.cluster import (KMeans, DBSCAN, MiniBatchKMeans,
                             SpectralClustering, MeanShift, AgglomerativeClustering,
                             Birch, OPTICS, SpectralBiclustering, SpectralCoclustering)
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding


CLUSTERING_KEY = 'clustering'
PRECLUST_TRANSFORM_KEY = 'cluster_preproc_transform'
CLUSTERING_ALGOS_KEY = 'algos'
CONSENSUS_KEY = 'consensus'
SELECTION_KEY = 'cluster_selection'
VISUALIZATION_KEY = 'cluster_visualization'


PRE_CLUSTER_TRANSFORM_REGISTRY = {

}

CLUSTERING_ALGOS_REGISTRY = {
    'KMeans': KMeans,
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
