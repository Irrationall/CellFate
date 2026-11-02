from __future__ import annotations
from anndata import AnnData
import numpy as np
import networkx as nx
import warnings
import scipy.sparse as sp
import scanpy as sc
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm
from functools import wraps
import logging
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def calc_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'⏳ Execution time for {func.__name__}: {end_time - start_time:.2f} seconds')
        return result
    return wrapper

def _threshold_sparse_matrix(matrix: sp.spmatrix, threshold: float) -> sp.csr_matrix:
    """Return a thresholded copy matching the previous sparse-matrix behavior."""
    temp_matrix = matrix.copy().tocsr()
    temp_matrix.data[temp_matrix.data < threshold] = 0
    temp_matrix.eliminate_zeros()
    return temp_matrix

def _thresholded_edge_values(matrix: sp.spmatrix) -> np.ndarray:
    """
    Collapse a sparse adjacency matrix into one value per undirected edge.

    A NetworkX Graph contains an edge after thresholding if either direction has
    a nonzero value above the threshold, so duplicate directed entries are
    represented by their maximum value.
    """
    coo = matrix.tocoo(copy=False)
    if coo.nnz == 0:
        return np.empty(0, dtype=matrix.dtype)
    nonzero = coo.data != 0
    if not np.any(nonzero):
        return np.empty(0, dtype=matrix.dtype)
    rows = coo.row[nonzero].astype(np.int64, copy=False)
    cols = coo.col[nonzero].astype(np.int64, copy=False)
    values = coo.data[nonzero]
    low = np.minimum(rows, cols)
    high = np.maximum(rows, cols)
    keys = low * matrix.shape[0] + high
    order = np.argsort(keys, kind='mergesort')
    keys = keys[order]
    values = values[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return np.sort(np.maximum.reduceat(values, starts))

def _thresholded_edge_count(edge_values: np.ndarray, threshold: float) -> int:
    return int(edge_values.size - np.searchsorted(edge_values, threshold, side='left'))

def _threshold_connected_original(matrix):
    """Return the final thresholded matrix without materializing NetworkX."""
    (low, high) = (0, 1)
    low_matrix = matrix
    low_matrix_is_thresholded = False
    while high - low > 0.0001:
        threshold = (low + high) / 2.0
        temp_matrix = _threshold_sparse_matrix(low_matrix, threshold)
        n_components = connected_components(temp_matrix, directed=False, return_labels=False)
        if n_components == 1:
            low = threshold
            low_matrix = temp_matrix
            low_matrix_is_thresholded = True
        else:
            high = threshold
    threshold = low
    if low_matrix_is_thresholded:
        matrix = low_matrix
    else:
        matrix = _threshold_sparse_matrix(matrix, threshold)
    return (matrix.tocsr(), threshold)

def _connectivity_threshold_original_mst(matrix):
    """
    Recover the original binary-search threshold from a maximum spanning tree.

    Connectivity at a threshold is determined by the minimum edge in a
    maximum spanning tree. Replaying the original scalar binary search against
    that bottleneck preserves its exact threshold quantization without
    repeatedly copying and filtering the sparse matrix.
    """
    matrix = matrix.tocsr()
    n_nodes = matrix.shape[0]
    if matrix.shape[1] != n_nodes:
        raise ValueError('matrix must be square')
    if n_nodes <= 1:
        bottleneck = 1.0
    else:
        adjacency = matrix.maximum(matrix.T).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        tree = minimum_spanning_tree(-adjacency)
        if tree.nnz != n_nodes - 1:
            raise ValueError('MIC candidate graph is disconnected')
        bottleneck = -float(np.max(tree.data))
    (low, high) = (0.0, 1.0)
    while high - low > 0.0001:
        threshold = (low + high) / 2.0
        if threshold <= bottleneck:
            low = threshold
        else:
            high = threshold
    return low

def _generate_sparse_graph_original(matrix):
    """Return an exact symmetric CSR graph and the original threshold."""
    (thresholded, threshold) = _threshold_connected_original(matrix)
    adjacency = thresholded.maximum(thresholded.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return (adjacency, threshold)

def _generate_graph_original(matrix):
    """
    Generate a graph using the "Original" method (ensure graph is connected).

    Parameters:
    - matrix: Sparse MIC matrix (CSR or CSC format)

    Returns:
    - graph: NetworkX graph
    - threshold: MIC threshold used
    """
    (matrix, threshold) = _threshold_connected_original(matrix)
    directed_graph = nx.from_scipy_sparse_array(matrix, create_using=nx.DiGraph)
    graph = directed_graph.to_undirected()
    return (graph, threshold)

def _generate_graph_multiples(matrix: sp.spmatrix, target_edgenum: int):
    """
    Generate a graph to match a target number of edges using the "Multiples" method.

    Parameters:
    - matrix: Sparse MIC matrix (CSR or CSC format)
    - target_edgenum: Target number of edges

    Returns:
    - graph: NetworkX graph
    - threshold: MIC threshold used
    """
    (low, high) = (0, 1)
    golden_ratio = (1 + 5 ** 0.5) / 2
    best_graph = None
    best_threshold = None
    best_diff = float('inf')
    edge_values = _thresholded_edge_values(matrix)
    with tqdm(total=float('inf'), desc='Constructing graph in progress...') as pbar:
        while high - low > 1e-05:
            mid1 = high - (high - low) / golden_ratio
            mid2 = low + (high - low) / golden_ratio
            diff1 = _get_graph_and_diff(edge_values, mid1, target_edgenum)
            diff2 = _get_graph_and_diff(edge_values, mid2, target_edgenum)
            if diff1 < best_diff:
                best_diff = diff1
                best_threshold = mid1
            if diff2 < best_diff:
                best_diff = diff2
                best_threshold = mid2
            if diff1 > diff2:
                low = mid1
            else:
                high = mid2
            pbar.update()
        pbar.close()
    best_graph = nx.from_scipy_sparse_array(_threshold_sparse_matrix(matrix, best_threshold))
    return (best_graph, best_threshold)

def _get_graph_and_diff(edge_values: np.ndarray, threshold: float, target_edgenum: int):
    """
    Apply a threshold to the MIC matrix, construct a graph, and compute the edge difference.

    Parameters:
    - edge_values: One threshold value per undirected edge
    - threshold: Threshold value to apply
    - target_edgenum: Target number of edges when multiply this number

    Returns:
    - diff: Absolute difference between the current and target number of edges
    """
    edge_num = _thresholded_edge_count(edge_values, threshold)
    diff = abs(edge_num - target_edgenum)
    return diff

@calc_time
def generate_graph(adata: AnnData, mic_key: str, method: str='Original', number_of_multiples: int=None):
    """
    Generate a graph from a sparse triangular MIC matrix.

    Parameters:
    - adata: name of AnnData object contain MIC matrix
    - mic_matrix: Sparse upper triangular MIC matrix (CSR format)
    - method: Method to generate the graph ("Original" or "Multiples")
    - number_of_multiples: Target edge count as a multiple of nodes (used only with "Multiples" method)

    Returns:
    - graph: Generated NetworkX graph
    - threshold: MIC threshold used to generate the graph
    """
    method_list = ['Original', 'Multiples']
    if mic_key not in adata.obsp:
        raise ValueError(f'{mic_key} is not present in `adata.obsp`. Please check the AnnData object.')
    if method not in method_list:
        raise ValueError(f"Invalid method '{method}'. Method should be one of {method_list}.")
    if sp.issparse(adata.obsp[mic_key]):
        matrix = adata.obsp[mic_key].copy()
    else:
        matrix = sp.csr_matrix(adata.obsp[mic_key].copy())
    if method == 'Original':
        (graph, threshold) = _generate_graph_original(matrix)
    elif method == 'Multiples':
        if number_of_multiples is None:
            raise ValueError("`number_of_multiples` must be specified for the 'Multiples' method.")
        target_edges = matrix.shape[0] * number_of_multiples
        logger.info('🎯 Target number of edges: %d', target_edges)
        (graph, threshold) = _generate_graph_multiples(matrix, target_edges)
        if nx.number_connected_components(graph) > 1:
            logger.warning('⚠️ The graph has more than one connected component.')
    logger.info('🛈 MIC threshold: %.6e', threshold)
    logger.info('🛈 Number of nodes: %d', len(graph.nodes()))
    logger.info('🛈 Number of edges: %d', len(graph.edges()))
    logger.info('🛈 Is graph connected: %s', nx.is_connected(graph))
    logger.info('🛈 Number of connected components: %d', nx.number_connected_components(graph))
    return (graph, threshold)
from typing import Literal, Tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
Weighting = Literal['adaptive_gaussian', 'cosine']
Symmetrize = Literal['max', 'mean', 'mutual']

def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding)
    if values.ndim != 2:
        raise ValueError('embedding must be a 2D cells x dimensions array')
    if values.shape[0] < 2:
        raise ValueError('embedding must contain at least two cells')
    if values.shape[1] < 1:
        raise ValueError('embedding must contain at least one dimension')
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError('embedding must contain numeric values')
    if not np.isfinite(values).all():
        raise ValueError('embedding contains NaN or infinite values')
    return np.asarray(values, dtype=np.float32, order='C')

def _symmetrize_adjacency(adjacency: sp.csr_matrix, method: Symmetrize) -> sp.csr_matrix:
    transpose = adjacency.T.tocsr()
    if method == 'max':
        result = adjacency.maximum(transpose)
    elif method == 'mean':
        result = (adjacency + transpose) * 0.5
    elif method == 'mutual':
        result = adjacency.minimum(transpose)
    else:
        raise ValueError('symmetrize must be one of: max, mean, mutual')
    result = result.tocsr()
    result.setdiag(0)
    result.eliminate_zeros()
    return result

def build_knn_adjacency(embedding: np.ndarray, n_neighbors: int=15, metric: str='cosine', weighting: Weighting='adaptive_gaussian', symmetrize: Symmetrize='max', n_jobs: int | None=None, min_weight: float=1e-08) -> sp.csr_matrix:
    """Construct a positive, symmetric sparse adjacency from cell embeddings.

    The adaptive Gaussian kernel uses each cell's farthest retained neighbor as
    its local scale. This avoids a single global bandwidth dominating dense or
    sparse regions of the embedding.
    """
    values = _validate_embedding(embedding)
    n_cells = values.shape[0]
    if not 1 <= int(n_neighbors) < n_cells:
        raise ValueError('n_neighbors must be between 1 and n_cells - 1')
    if min_weight < 0:
        raise ValueError('min_weight must be non-negative')
    if weighting == 'cosine' and metric != 'cosine':
        raise ValueError("cosine weighting requires metric='cosine'")
    model = NearestNeighbors(n_neighbors=int(n_neighbors) + 1, metric=metric, algorithm='brute' if metric == 'cosine' else 'auto', n_jobs=n_jobs)
    model.fit(values)
    (distances, indices) = model.kneighbors(values, return_distance=True)
    neighbor_indices = np.empty((n_cells, int(n_neighbors)), dtype=np.int64)
    neighbor_distances = np.empty((n_cells, int(n_neighbors)), dtype=np.float64)
    for row in range(n_cells):
        keep = indices[row] != row
        row_indices = indices[row, keep][:int(n_neighbors)]
        row_distances = distances[row, keep][:int(n_neighbors)]
        if row_indices.size != int(n_neighbors):
            raise RuntimeError('nearest-neighbor search did not return enough non-self cells')
        neighbor_indices[row] = row_indices
        neighbor_distances[row] = row_distances
    if weighting == 'adaptive_gaussian':
        local_scale = neighbor_distances[:, -1]
        positive = local_scale[local_scale > 0]
        fallback = float(np.median(positive)) if positive.size else 1.0
        local_scale = np.where(local_scale > 0, local_scale, fallback)
        weights = np.exp(-np.square(neighbor_distances / local_scale[:, None]))
    elif weighting == 'cosine':
        weights = np.clip(1.0 - neighbor_distances, 0.0, 1.0)
    else:
        raise ValueError('weighting must be one of: adaptive_gaussian, cosine')
    if min_weight > 0:
        weights = np.maximum(weights, min_weight)
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), int(n_neighbors))
    adjacency = sp.csr_matrix((weights.ravel(), (rows, neighbor_indices.ravel())), shape=(n_cells, n_cells), dtype=np.float64)
    return _symmetrize_adjacency(adjacency, symmetrize)

def build_knn_graph(embedding: np.ndarray, n_neighbors: int=15, metric: str='cosine', weighting: Weighting='adaptive_gaussian', symmetrize: Symmetrize='max', n_jobs: int | None=None) -> Tuple[nx.Graph, sp.csr_matrix]:
    """Return a NetworkX graph and its sparse adjacency."""
    adjacency = build_knn_adjacency(embedding, n_neighbors=n_neighbors, metric=metric, weighting=weighting, symmetrize=symmetrize, n_jobs=n_jobs)
    graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.Graph)
    return (graph, adjacency)

def count_components(adjacency: sp.spmatrix) -> int:
    """Return the number of undirected connected components."""
    return int(connected_components(adjacency, directed=False, return_labels=False))
import time
from typing import Literal, Sequence
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from .mic import calculate_MIC, calculate_MIC_candidate, calculate_MIC_filtered
from .rdim import run_selected_sources
from .stats import FatePredictor
NeighborMethod = Literal['auto', 'exact', 'pynndescent']
ExactGraphStrategy = Literal['auto', 'full', 'filtered']

def _upper_pairs_from_neighbors(neighbor_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_cells = neighbor_indices.shape[0]
    rows = np.repeat(np.arange(num_cells, dtype=np.int64), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1).astype(np.int64, copy=False)
    low = np.minimum(rows, cols)
    high = np.maximum(rows, cols)
    valid = low != high
    keys = np.unique(low[valid] * num_cells + high[valid])
    return (keys // num_cells, keys % num_cells)

def build_neighbor_candidate_pairs(embedding, n_neighbors: int=600, metric: str='euclidean', n_jobs: int=1, method: NeighborMethod='auto', random_state: int=0, exact_max_cells: int=20000) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse candidate pairs used only for a certified MIC lower bound."""
    values = np.asarray(embedding, dtype=np.float32, order='C')
    if values.ndim != 2:
        raise ValueError('embedding must be a 2D cells x dimensions array')
    num_cells = values.shape[0]
    n_neighbors = int(n_neighbors)
    if not 1 <= n_neighbors < num_cells:
        raise ValueError('n_neighbors must be between 1 and n_cells - 1')
    if method == 'auto':
        method = 'exact' if num_cells <= exact_max_cells else 'pynndescent'
    if method == 'exact':
        model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=n_jobs).fit(values)
        raw_indices = model.kneighbors(values, return_distance=False)
    elif method == 'pynndescent':
        try:
            from pynndescent import NNDescent
        except ImportError as error:
            raise ImportError('pynndescent is required for approximate neighbor candidates') from error
        index = NNDescent(values, n_neighbors=n_neighbors + 1, metric=metric, n_jobs=n_jobs, random_state=random_state)
        raw_indices = index.neighbor_graph[0]
    else:
        raise ValueError("method must be 'auto', 'exact', or 'pynndescent'")
    neighbor_indices = np.empty((num_cells, n_neighbors), dtype=np.int64)
    for row in range(num_cells):
        row_indices = raw_indices[row]
        retained = row_indices[row_indices != row][:n_neighbors]
        if retained.size != n_neighbors:
            raise RuntimeError('neighbor search did not return enough non-self cells')
        neighbor_indices[row] = retained
    return _upper_pairs_from_neighbors(neighbor_indices)

def calculate_exact_mic_graph(data, candidate_pairs=None, candidate_neighbors: int=600, neighbor_method: NeighborMethod='auto', neighbor_metric: str='euclidean', n_threads: int=1, random_state: int=0, execution_strategy: ExactGraphStrategy='auto', full_matrix_max_cells: int=8000):
    """
    Recover the exact CellOrigin MIC graph without retaining all pair scores.

    Small inputs use the faster full upper-triangular MIC path. Larger inputs
    use candidate edges only to establish a certified lower bound, then still
    score every upper-triangular pair with the exact active minepy estimator.
    """
    values = np.asarray(data, dtype=np.float64, order='C')
    if values.ndim != 2:
        raise ValueError('data must be a 2D cells x features array')
    if execution_strategy not in {'auto', 'full', 'filtered'}:
        raise ValueError("execution_strategy must be 'auto', 'full', or 'filtered'")
    if full_matrix_max_cells < 2:
        raise ValueError('full_matrix_max_cells must be at least 2')
    strategy = execution_strategy
    if strategy == 'auto':
        strategy = 'full' if candidate_pairs is None and values.shape[0] <= full_matrix_max_cells else 'filtered'
    timings = {}
    if strategy == 'full':
        started = time.perf_counter()
        full_matrix = calculate_MIC(values, num_processes=n_threads).tocsr()
        timings['all_pairs_mic_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        (graph, threshold) = _generate_sparse_graph_original(full_matrix)
        final_matrix = _threshold_sparse_matrix(full_matrix, threshold)
        timings['final_graph_seconds'] = time.perf_counter() - started
        timings['total_seconds'] = sum(timings.values())
        metadata = {'cells': int(values.shape[0]), 'features': int(values.shape[1]), 'execution_strategy': strategy, 'full_pairs': int(full_matrix.nnz), 'final_threshold': float(threshold), 'final_edges': int(final_matrix.nnz), 'timings': timings}
        return (graph, final_matrix, metadata)
    started = time.perf_counter()
    if candidate_pairs is None:
        candidate_pairs = build_neighbor_candidate_pairs(values, n_neighbors=candidate_neighbors, metric=neighbor_metric, n_jobs=n_threads, method=neighbor_method, random_state=random_state)
    timings['candidate_search_seconds'] = time.perf_counter() - started
    started = time.perf_counter()
    candidate_matrix = calculate_MIC_candidate(values, candidate_pairs, num_processes=n_threads).tocsr()
    candidate_matrix.eliminate_zeros()
    try:
        lower_bound = _connectivity_threshold_original_mst(candidate_matrix)
    except ValueError as error:
        raise ValueError('candidate MIC graph is disconnected; increase candidate_neighbors') from error
    timings['candidate_mic_and_bound_seconds'] = time.perf_counter() - started
    started = time.perf_counter()
    filtered_matrix = calculate_MIC_filtered(values, min_mic=lower_bound, num_processes=n_threads)
    timings['all_pairs_filtered_mic_seconds'] = time.perf_counter() - started
    started = time.perf_counter()
    (graph, threshold) = _generate_sparse_graph_original(filtered_matrix)
    final_matrix = _threshold_sparse_matrix(filtered_matrix, threshold)
    timings['final_graph_seconds'] = time.perf_counter() - started
    timings['total_seconds'] = sum(timings.values())
    metadata = {'cells': int(values.shape[0]), 'features': int(values.shape[1]), 'execution_strategy': strategy, 'candidate_pairs': int(candidate_matrix.nnz), 'certified_lower_bound': float(lower_bound), 'filtered_pairs': int(filtered_matrix.nnz), 'final_threshold': float(threshold), 'final_edges': int(final_matrix.nnz), 'timings': timings}
    return (graph, final_matrix, metadata)

def run_exact_fate_pipeline(adata, data, group_by: str, source: str, target: Sequence[str], times, candidate_pairs=None, candidate_neighbors: int=600, neighbor_method: NeighborMethod='auto', neighbor_metric: str='euclidean', n_threads: int=1, rd_workers: int=1, use_spectral_gap: bool=False, fate_row_batch_size: int=256, disable_tqdm: bool=False):
    """Run exact MIC graph, selected strict RD, and bounded-memory fate."""
    total_started = time.perf_counter()
    (graph, mic_matrix, graph_metadata) = calculate_exact_mic_graph(data, candidate_pairs=candidate_pairs, candidate_neighbors=candidate_neighbors, neighbor_method=neighbor_method, neighbor_metric=neighbor_metric, n_threads=n_threads)
    labels = adata.obs[group_by]
    source_indices = np.flatnonzero(labels.to_numpy() == source)
    target_indices = np.flatnonzero(labels.isin(target).to_numpy())
    if not source_indices.size:
        raise ValueError(f'source cluster {source!r} has no cells')
    if not target_indices.size:
        raise ValueError('target clusters have no cells')
    started = time.perf_counter()
    (relative_dimensions, peak_times) = run_selected_sources(graph, times, source_nodes=target_indices, retain_nodes=source_indices, use_spectral_gap=use_spectral_gap, disable_tqdm=disable_tqdm, n_workers=rd_workers)
    rd_seconds = time.perf_counter() - started
    predictor = FatePredictor.from_selected_rd(adata, relative_dimensions, group_by=group_by, source=source, target=list(target), source_indices=source_indices, target_indices=target_indices)
    started = time.perf_counter()
    predictor.predict_fate_streaming(resolve_tie=True, na_value='max', use_fdr=True, res_to_string=True, row_batch_size=fate_row_batch_size, disable_tqdm=disable_tqdm)
    fate_seconds = time.perf_counter() - started
    metadata = {'graph': graph_metadata, 'rd': {'source_rows': int(source_indices.size), 'target_sources': int(target_indices.size), 'nnz': int(relative_dimensions.nnz), 'seconds': float(rd_seconds)}, 'fate': {'row_batch_size': int(fate_row_batch_size), 'seconds': float(fate_seconds)}, 'total_seconds': float(time.perf_counter() - total_started)}
    return {'graph': graph, 'mic_matrix': mic_matrix, 'relative_dimensions': relative_dimensions, 'peak_times': peak_times, 'fate': predictor.reld_adata.obs.copy(), 'source_indices': source_indices, 'target_indices': target_indices, 'metadata': metadata}
