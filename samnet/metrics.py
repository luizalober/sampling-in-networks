import networkx as nx
import numpy as np
import scipy.linalg as la
from .statistics import information_metrics, moments
from scipy.linalg.lapack import dpotrf, dpotri
import scipy.spatial.distance as sp_dist
#from scipy.sparse.linalg import eigsh


#Can't run this. Resistance distance scales badly with N

#def get_resistance_and_spectral_distances(graph, *args, **kwargs):
    #Only one laplacian calculation (O(N³))
#    L_sparse = nx.laplacian_matrix(graph).astype(float)
#    degrees = L_sparse.diagonal()
#    L_dense = L_sparse.toarray()

    #Spectral distance
#    D2_spec = degrees[:, None] + degrees[None, :] - 2.0 * L_dense
#    np.maximum(D2_spec, 0, out=D2_spec)
#    D_spectral = np.sqrt(D2_spec, out=D2_spec)

    #Resistance distance (with Moore-Penrose pseudoinverse)
    #eigenvalues, eigenvectors = la.eigh(L_dense)
    # = eigenvalues > 1e-10
    #vals = eigenvalues[valid_mask]
    #vecs = eigenvectors[:, valid_mask]
    #Y = vecs / np.sqrt(vals)

    #condensed_resist = sp_dist.pdist(Y, metric='sqeuclidean')
    #D_resist = sp_dist.squareform(condensed_resist)


    #Approx. of resistance distance

#    vals, vecs = eigsh(L_sparse, k=100+1, which='SM')
#    vals = vals[1:]
#    vecs = vecs[:, 1:]
#    Y = vecs / np.sqrt(vals)
#    condensed_resist = sp_dist.pdist(Y, metric='sqeuclidean')
#    D_resist = sp_dist.squareform(condensed_resist)


#    return D_spectral, D_resist


#Just spectral distance
def get_spectral_distance(graph, *args, **kwargs):
    #Only one laplacian calculation (O(N³))
    L = nx.laplacian_matrix(graph).toarray().astype(float)
    N = L.shape[0]

    #Spectral distance
    diag_L = np.diag(L)
    D2_spec = diag_L[:, None] + diag_L[None, :] - 2.0 * L
    np.maximum(D2_spec, 0, out=D2_spec)
    D_spectral = np.sqrt(D2_spec, out=D2_spec)

    return D_spectral


# === Global metrics ===

def assortativity(graph, *args, **kwargs):
    return nx.degree_assortativity_coefficient(graph)


def transitivity(graph, *args, **kwargs):
    return nx.transitivity(graph)


def avg_spl(graph, *args, **kwargs):
    try:
        return nx.average_shortest_path_length(graph)
    except nx.NetworkXError:
        return np.nan


def complexity_coefficient(graph, *args, **kwargs):
    degree_list = list(dict(graph.degree).values())
    m1 = moments(degree_list, 1)
    return moments(degree_list, 2) / m1 if m1 != 0 else 0.0


# Local metrics

def degree_info(graph, shannon_bin=1, n_moments=1, *args, **kwargs):
    return information_metrics(dict(graph.degree).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def clustering_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    return information_metrics(nx.clustering(graph).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def betweenness_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    return information_metrics(nx.betweenness_centrality(graph).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def closeness_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    return information_metrics(nx.closeness_centrality(graph).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def kcore_info(graph, shannon_bin=1, n_moments=1, *args, **kwargs):
    return information_metrics(nx.core_number(graph).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def pagerank_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    return information_metrics(nx.pagerank(graph).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def eigenvector_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    return information_metrics(nx.eigenvector_centrality(graph, max_iter=30000).values(), shannon_bin=shannon_bin, n_moments=n_moments)


def global_efficiency(graph, *args, **kwargs):
    return nx.global_efficiency(graph)


def local_efficiency(graph, *args, **kwargs):
    return nx.local_efficiency(graph)


def spectral_distance_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    D_spec = get_spectral_distance(graph)
    vals = D_spec.ravel()
    return {
        'max': float(D_spec.max()),
        'mean': float(D_spec.mean()),
        **information_metrics(vals, shannon_bin=shannon_bin, n_moments=n_moments)
    }

'''
def spectral_and_resistance_distance_info(graph, shannon_bin=0.1, n_moments=1, *args, **kwargs):
    D_spec, D_res = get_resistance_and_spectral_distances(graph)

    results = {}
    distancias = {
        'spectral': D_spec,
        'resistance': D_res
    }

    for prefix, D in distancias.items():
        vals = D.ravel()

        results[f'{prefix}_max'] = float(D.max())
        results[f'{prefix}_mean'] = float(D.mean())
        info = information_metrics(vals, shannon_bin=shannon_bin, n_moments=n_moments)
        for k, v in info.items():
            results[f'{prefix}_{k}'] = v

    return results
'''
