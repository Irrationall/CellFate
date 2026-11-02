"""Main functions for calculating relative dimensions in a graph."""
import os
import sys
import tempfile
import time
import warnings
import logging
from anndata import AnnData
from functools import partial, wraps, reduce
from operator import index as integer_index, add
from multiprocessing import Pool, cpu_count
import networkx as nx
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm
from cellfate import utils as _strict_rd_worker
try:
    from scipy.sparse.linalg import _expm_multiply as _scipy_expm
except ImportError:
    _scipy_expm = None
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
PRECISION = 1e-08
SGAP_THRESHOLD = 0.001
CNVG_THRESHOLD = 1e-06
AUTO_MAX_WORKERS = 16
STRICT_RD_BATCHES_PER_WORKER = 32
_STRICT_RD_WORKER_STATE = None
_STRICT_RD_TEMP_DIRECTORIES = []
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def _ordered_sum(values):
    # Python 3.12 changed float sum; preserve the original RD accumulation order.
    return sum(values) if sys.version_info < (3, 12) else reduce(add, values, 0)

def _weighted_degree_exact(graph, node):
    if sys.version_info < (3, 12):
        return graph.degree(node, weight='weight')
    def adjacent_degree(neighbors):
        if graph.is_multigraph():
            return _ordered_sum(data.get('weight', 1) for edges in neighbors.values() for data in edges.values())
        return _ordered_sum(data.get('weight', 1) for data in neighbors.values())
    if graph.is_directed():
        return adjacent_degree(graph.succ[node]) + adjacent_degree(graph.pred[node])
    neighbors = graph.adj[node]
    degree = adjacent_degree(neighbors)
    if node in neighbors:
        degree += adjacent_degree({node: neighbors[node]})
    return degree

def calc_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info("⏳ Function '%s' executed in %.2f seconds", func.__name__, end_time - start_time)
        return result
    return wrapper

def run_all_sources(graph, times, use_spectral_gap=True, disable_tqdm=False, batch_size=100, n_workers=1, cache_expm_plans=True, strict_block_size=1, strict_block_backend='scipy'):
    """
    Compute relative dimensions of all nodes in a graph.

    Parameters:
    - graph: NetworkX graph
    - times: Array of time steps for diffusion
    - use_spectral_gap: Whether to normalize Laplacian by spectral gap
    - disable_tqdm: Disable progress bar
    - batch_size: Number of nodes to process in each batch
    - n_workers: Number of workers for parallel processing
    - cache_expm_plans: Reuse exact expm setup for strict single-source batches
    - strict_block_size: Opt-in source-major blocks using scalar expm plans and
      per-source convergence. Requires batch_size=1 and cached plans. Default
      1 preserves the existing scalar path; 16 is benchmarked on Rewind.
    - strict_block_backend: 'scipy' (default) or opt-in 'numba' buffered CSR
      products. The latter requires strict_block_size > 1.

    Returns:
    - relative_dimensions: Array of relative dimensions
    - peak_times: Array of peak times
    """
    try:
        strict_block_size = integer_index(strict_block_size)
    except TypeError as error:
        raise ValueError('strict_block_size must be a positive integer') from error
    if strict_block_size < 1:
        raise ValueError('strict_block_size must be a positive integer')
    if strict_block_size > 1 and (batch_size != 1 or not cache_expm_plans):
        raise ValueError('strict source blocks require batch_size=1 and cache_expm_plans=True')
    if strict_block_backend not in ('scipy', 'numba', 'register'):
        raise ValueError("strict_block_backend must be 'scipy', 'numba', or 'register'")
    if strict_block_backend in ('numba', 'register') and strict_block_size == 1:
        raise ValueError('compiled block backend requires strict_block_size > 1')
    if strict_block_backend == 'register' and strict_block_size != 16:
        raise ValueError('experimental register backend requires strict_block_size == 16')
    if n_workers is None:
        n_workers = min(cpu_count(), AUTO_MAX_WORKERS)
    n_workers = max(1, int(n_workers))
    degrees = np.array([_weighted_degree_exact(graph, node) for node in graph])
    total_degree = degrees.sum()
    len_graph = len(graph)
    constant = total_degree / len_graph
    initial_values = constant / degrees
    (laplacian, spectral_gap) = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)
    relative_dimensions = np.full((len_graph, len_graph), np.nan)
    peak_times = np.full((len_graph, len_graph), np.nan)
    if batch_size == 1 and cache_expm_plans:
        plans = _prepare_strict_expm_plans(laplacian, times)
        plans_valid = plans is not None and _validate_strict_expm_plans(laplacian, initial_values, times, plans)
        if plans_valid and strict_block_size > 1:
            from cellfate.utils import block_validate_plans
            materialized = [_materialize_strict_expm_plan(*plan) for plan in plans]
            if not block_validate_plans(materialized, initial_values, strict_block_size, strict_block_backend):
                logger.warning('Strict source-block validation failed; using scalar cached RD.')
                strict_block_size = 1
                strict_block_backend = 'scipy'
        if plans_valid:
            _run_cached_strict_sources(plans, initial_values, times, spectral_gap, relative_dimensions, peak_times, n_workers, disable_tqdm, strict_block_size=strict_block_size, strict_block_backend=strict_block_backend)
            np.fill_diagonal(relative_dimensions, np.nan)
            np.fill_diagonal(peak_times, np.nan)
            logger.info('✅ All sources processed.')
            return (relative_dimensions, peak_times)
    batches = [(laplacian, initial_values, times, spectral_gap, i, min(i + batch_size, len_graph), len_graph) for i in range(0, len_graph, batch_size)]
    if n_workers == 1:
        results = [process_batch(batch) for batch in tqdm(batches, disable=disable_tqdm)]
    else:
        results = Parallel(n_jobs=n_workers, prefer='processes')((delayed(process_batch)(batch) for batch in tqdm(batches, disable=disable_tqdm)))
    for (rd_batch, pt_batch, start_idx) in results:
        end_idx = start_idx + rd_batch.shape[1]
        relative_dimensions[:, start_idx:end_idx] = rd_batch
        peak_times[:, start_idx:end_idx] = pt_batch
    np.fill_diagonal(relative_dimensions, np.nan)
    np.fill_diagonal(peak_times, np.nan)
    logger.info('✅ All sources processed.')
    return (relative_dimensions, peak_times)

def _normalize_node_indices(indices, num_nodes, name):
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional sequence')
    if values.size and (np.any(values < 0) or np.any(values >= num_nodes)):
        raise ValueError(f'{name} contains an out-of-range node index')
    if np.unique(values).size != values.size:
        raise ValueError(f'{name} must not contain duplicate node indices')
    return values

def _sparse_weighted_degrees_exact(adjacency):
    """Match NetworkX Graph.degree's ordered Python summation."""
    adjacency = adjacency.tocsr()
    return np.asarray([_ordered_sum((float(value) for value in adjacency.data[adjacency.indptr[row]:adjacency.indptr[row + 1]])) for row in range(adjacency.shape[0])], dtype=np.float64)

def run_selected_sources(graph, times, source_nodes, retain_nodes=None, use_spectral_gap=True, spectral_gap_override=None, disable_tqdm=False, n_workers=1, cache_expm_plans=True):
    """
    Compute exact strict RD columns for selected diffusion source nodes.

    Rows are returned in ``retain_nodes`` order and columns in ``source_nodes``
    order. Missing or unreachable values are represented as structural zeros in
    the returned CSR matrices, matching a sparse slice of ``run_all_sources``.
    """
    num_nodes = graph.shape[0] if sp.issparse(graph) else len(graph)
    source_nodes = _normalize_node_indices(source_nodes, num_nodes, 'source_nodes')
    if retain_nodes is None:
        retain_nodes = np.arange(num_nodes, dtype=np.int64)
    else:
        retain_nodes = _normalize_node_indices(retain_nodes, num_nodes, 'retain_nodes')
    if n_workers is None:
        n_workers = min(cpu_count(), AUTO_MAX_WORKERS)
    n_workers = max(1, int(n_workers))
    output_shape = (retain_nodes.size, source_nodes.size)
    if not source_nodes.size or not retain_nodes.size:
        empty = sp.csr_matrix(output_shape, dtype=np.float64)
        return (empty, empty.copy())
    if sp.issparse(graph):
        degrees = _sparse_weighted_degrees_exact(graph)
    else:
        degrees = np.array([_weighted_degree_exact(graph, node) for node in graph])
    total_degree = degrees.sum()
    initial_values = total_degree / num_nodes / degrees
    if spectral_gap_override is None:
        (laplacian, spectral_gap) = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)
    else:
        spectral_gap = float(spectral_gap_override)
        if not np.isfinite(spectral_gap) or spectral_gap <= 0:
            raise ValueError('spectral_gap_override must be finite and positive')
        (laplacian, _) = construct_laplacian(graph, use_spectral_gap=False)
        laplacian /= spectral_gap
    output_row_lookup = np.full(num_nodes, -1, dtype=np.int64)
    output_row_lookup[retain_nodes] = np.arange(retain_nodes.size, dtype=np.int64)
    output_column_lookup = np.full(num_nodes, -1, dtype=np.int64)
    output_column_lookup[source_nodes] = np.arange(source_nodes.size, dtype=np.int64)
    rd_row_parts = []
    rd_column_parts = []
    rd_value_parts = []
    pt_row_parts = []
    pt_column_parts = []
    pt_value_parts = []

    def collect(packed_result):
        (source_batch, rd_indptr, rd_rows, rd_values, pt_indptr, pt_rows, pt_values) = packed_result
        rd_row_parts.append(rd_rows)
        rd_column_parts.append(np.repeat(output_column_lookup[source_batch], np.diff(rd_indptr)))
        rd_value_parts.append(rd_values)
        pt_row_parts.append(pt_rows)
        pt_column_parts.append(np.repeat(output_column_lookup[source_batch], np.diff(pt_indptr)))
        pt_value_parts.append(pt_values)
    plans = None
    if cache_expm_plans:
        plans = _prepare_strict_expm_plans(laplacian, times)
        if plans is not None and (not _validate_strict_expm_plans(laplacian, initial_values, times, plans)):
            plans = None
    if plans is not None:
        initializer_args = (_pack_strict_expm_plans(plans), initial_values, np.asarray(times), spectral_gap, output_row_lookup)
        batch_size = max(1, (source_nodes.size + n_workers * STRICT_RD_BATCHES_PER_WORKER - 1) // (n_workers * STRICT_RD_BATCHES_PER_WORKER))
        source_batches = [source_nodes[start:start + batch_size] for start in range(0, source_nodes.size, batch_size)]
        if n_workers == 1:
            _strict_rd_worker.worker_initialize(*initializer_args)
            results = (_strict_rd_worker.worker_process_source_batch_packed(source_batch) for source_batch in tqdm(source_batches, disable=disable_tqdm))
        else:
            temp_directory = tempfile.TemporaryDirectory(prefix='cellorigin-selected-rd-plans-')
            _STRICT_RD_TEMP_DIRECTORIES.append(temp_directory)
            plan_path = os.path.join(temp_directory.name, 'strict_expm_plans.joblib')
            dump(initializer_args, plan_path, compress=0)
            results = Parallel(n_jobs=n_workers, prefer='processes', return_as='generator_unordered', batch_size='auto', initializer=_strict_rd_worker.worker_initialize_from_file, initargs=(plan_path,))((delayed(_strict_rd_worker.worker_process_source_batch_packed)(source_batch) for source_batch in tqdm(source_batches, disable=disable_tqdm)))
        for result in results:
            collect(result)
    else:
        for source_idx in tqdm(source_nodes, disable=disable_tqdm):
            source = np.zeros(num_nodes, dtype=np.float64)
            source[source_idx] = initial_values[source_idx]
            (peak_amplitudes, peak_pos) = compute_node_trajectory_peaks(laplacian, source, times)
            (rd_values, pt_values, _, _) = _extract_relative_dimensions_from_peaks(np.asarray(times), peak_amplitudes[:, None], peak_pos[:, None], source[:, None], spectral_gap)
            rd_source = rd_values[retain_nodes, 0]
            pt_source = pt_values[retain_nodes, 0]
            rd_rows = np.flatnonzero(~np.isnan(rd_source))
            pt_rows = np.flatnonzero(~np.isnan(pt_source))
            collect((np.asarray([source_idx], dtype=np.int64), np.asarray([0, rd_rows.size], dtype=np.int64), rd_rows, rd_source[rd_rows], np.asarray([0, pt_rows.size], dtype=np.int64), pt_rows, pt_source[pt_rows]))

    def make_sparse(row_parts, column_parts, value_parts):
        if not value_parts:
            return sp.csr_matrix(output_shape, dtype=np.float64)
        matrix = sp.csr_matrix((np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(column_parts))), shape=output_shape)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix
    relative_dimensions = make_sparse(rd_row_parts, rd_column_parts, rd_value_parts)
    peak_times = make_sparse(pt_row_parts, pt_column_parts, pt_value_parts)
    logger.info('✅ Selected strict RD processed: %d retained rows × %d sources.', retain_nodes.size, source_nodes.size)
    return (relative_dimensions, peak_times)

def run_selected_sources_lazy_tiled(graph, times, source_nodes, retain_nodes=None, use_spectral_gap=True, spectral_gap_override=None, tile_size=8, n_threads=1):
    """Compute bitwise-exact selected strict RD with one expm plan in memory."""
    if not sp.issparse(graph):
        raise TypeError('lazy tiled strict RD requires a SciPy sparse graph')
    from cellfate.utils import lazy_tiled_compute_selected_peaks_lazy_tiled
    num_nodes = graph.shape[0]
    source_nodes = _normalize_node_indices(source_nodes, num_nodes, 'source_nodes')
    if retain_nodes is None:
        retain_nodes = np.arange(num_nodes, dtype=np.int64)
    else:
        retain_nodes = _normalize_node_indices(retain_nodes, num_nodes, 'retain_nodes')
    output_shape = (retain_nodes.size, source_nodes.size)
    if not source_nodes.size or not retain_nodes.size:
        empty = sp.csr_matrix(output_shape, dtype=np.float64)
        return (empty, empty.copy(), [])
    degrees = _sparse_weighted_degrees_exact(graph)
    initial_values = degrees.sum() / num_nodes / degrees
    if spectral_gap_override is None:
        (laplacian, spectral_gap) = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)
    else:
        spectral_gap = float(spectral_gap_override)
        if not np.isfinite(spectral_gap) or spectral_gap <= 0:
            raise ValueError('spectral_gap_override must be finite and positive')
        (laplacian, _) = construct_laplacian(graph, use_spectral_gap=False)
        laplacian /= spectral_gap
    (peak_amplitudes, peak_pos, plan_metadata) = lazy_tiled_compute_selected_peaks_lazy_tiled(laplacian, initial_values, np.asarray(times), source_nodes, retain_nodes, tile_size=max(1, int(tile_size)), n_threads=min(32, max(1, int(n_threads))))
    initial_selected = np.zeros(output_shape, dtype=np.float64)
    retain_lookup = np.full(num_nodes, -1, dtype=np.int64)
    retain_lookup[retain_nodes] = np.arange(retain_nodes.size, dtype=np.int64)
    retained_source_rows = retain_lookup[source_nodes]
    present = retained_source_rows >= 0
    initial_selected[retained_source_rows[present], np.flatnonzero(present)] = initial_values[source_nodes[present]]
    peak_times = np.asarray(times)[peak_pos]
    unreachable = peak_amplitudes < 1.0 / num_nodes + PRECISION
    initial_nonzero = initial_selected > 0
    missed_peaks = (peak_pos == 0) & ~initial_nonzero
    if np.any(missed_peaks):
        warnings.warn('Some peaks are not detected because they occur at the minimum time step. Consider reducing the minimum time step in `times` to capture these peaks.')
    peak_times[unreachable] = np.nan
    peak_amplitudes[unreachable] = np.nan
    peak_times[initial_nonzero] = np.nan
    peak_amplitudes[initial_nonzero] = np.nan
    diffusion_coefficient = 0.5 / spectral_gap
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_dimensions = -2.0 * np.log(peak_amplitudes) / (1.0 + np.log(peak_times) + np.log(4.0 * diffusion_coefficient * np.pi))
    relative_dimensions[relative_dimensions < 0] = np.nan
    rd = sp.csr_matrix(np.nan_to_num(relative_dimensions, nan=0.0))
    pt = sp.csr_matrix(np.nan_to_num(peak_times, nan=0.0))
    rd.eliminate_zeros()
    pt.eliminate_zeros()
    rd.sort_indices()
    pt.sort_indices()
    logger.info('✅ Lazy tiled selected strict RD processed: %d retained rows × %d sources.', retain_nodes.size, source_nodes.size)
    return (rd, pt, plan_metadata)

def run_selected_sources_lazy_threaded(graph, times, source_nodes, retain_nodes=None, use_spectral_gap=True, spectral_gap_override=None, n_threads=1):
    """Compute bitwise-exact selected strict RD with shared SciPy plans."""
    if not sp.issparse(graph):
        raise TypeError('lazy threaded strict RD requires a SciPy sparse graph')
    from cellfate.utils import lazy_threaded_compute_selected_peaks_lazy_threaded
    num_nodes = graph.shape[0]
    source_nodes = _normalize_node_indices(source_nodes, num_nodes, 'source_nodes')
    if retain_nodes is None:
        retain_nodes = np.arange(num_nodes, dtype=np.int64)
    else:
        retain_nodes = _normalize_node_indices(retain_nodes, num_nodes, 'retain_nodes')
    output_shape = (retain_nodes.size, source_nodes.size)
    if not source_nodes.size or not retain_nodes.size:
        empty = sp.csr_matrix(output_shape, dtype=np.float64)
        return (empty, empty.copy(), [])
    degrees = _sparse_weighted_degrees_exact(graph)
    initial_values = degrees.sum() / num_nodes / degrees
    if spectral_gap_override is None:
        (laplacian, spectral_gap) = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)
    else:
        spectral_gap = float(spectral_gap_override)
        if not np.isfinite(spectral_gap) or spectral_gap <= 0:
            raise ValueError('spectral_gap_override must be finite and positive')
        (laplacian, _) = construct_laplacian(graph, use_spectral_gap=False)
        laplacian /= spectral_gap
    (peak_amplitudes, peak_pos, plan_metadata) = lazy_threaded_compute_selected_peaks_lazy_threaded(laplacian, initial_values, np.asarray(times), source_nodes, retain_nodes, n_threads=min(32, max(1, int(n_threads))))
    initial_selected = np.zeros(output_shape, dtype=np.float64)
    retain_lookup = np.full(num_nodes, -1, dtype=np.int64)
    retain_lookup[retain_nodes] = np.arange(retain_nodes.size, dtype=np.int64)
    retained_source_rows = retain_lookup[source_nodes]
    present = retained_source_rows >= 0
    initial_selected[retained_source_rows[present], np.flatnonzero(present)] = initial_values[source_nodes[present]]
    peak_times = np.asarray(times)[peak_pos]
    unreachable = peak_amplitudes < 1.0 / num_nodes + PRECISION
    initial_nonzero = initial_selected > 0
    missed_peaks = (peak_pos == 0) & ~initial_nonzero
    if np.any(missed_peaks):
        warnings.warn('Some peaks are not detected because they occur at the minimum time step. Consider reducing the minimum time step in `times` to capture these peaks.')
    peak_times[unreachable] = np.nan
    peak_amplitudes[unreachable] = np.nan
    peak_times[initial_nonzero] = np.nan
    peak_amplitudes[initial_nonzero] = np.nan
    diffusion_coefficient = 0.5 / spectral_gap
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_dimensions = -2.0 * np.log(peak_amplitudes) / (1.0 + np.log(peak_times) + np.log(4.0 * diffusion_coefficient * np.pi))
    relative_dimensions[relative_dimensions < 0] = np.nan
    rd = sp.csr_matrix(np.nan_to_num(relative_dimensions, nan=0.0))
    pt = sp.csr_matrix(np.nan_to_num(peak_times, nan=0.0))
    rd.eliminate_zeros()
    pt.eliminate_zeros()
    rd.sort_indices()
    pt.sort_indices()
    logger.info('✅ Lazy threaded selected strict RD processed: %d retained rows × %d sources.', retain_nodes.size, source_nodes.size)
    return (rd, pt, plan_metadata)

def run_selected_sources_lazy_scipy_tiled(graph, times, source_nodes, retain_nodes=None, use_spectral_gap=True, spectral_gap_override=None, tile_size=8, n_threads=1):
    """Compute bitwise-exact strict RD with threaded SciPy source tiles."""
    if not sp.issparse(graph):
        raise TypeError('lazy SciPy-tiled strict RD requires a SciPy sparse graph')
    from cellfate.utils import lazy_scipy_tiled_compute_selected_peaks_lazy_scipy_tiled
    num_nodes = graph.shape[0]
    source_nodes = _normalize_node_indices(source_nodes, num_nodes, 'source_nodes')
    if retain_nodes is None:
        retain_nodes = np.arange(num_nodes, dtype=np.int64)
    else:
        retain_nodes = _normalize_node_indices(retain_nodes, num_nodes, 'retain_nodes')
    output_shape = (retain_nodes.size, source_nodes.size)
    if not source_nodes.size or not retain_nodes.size:
        empty = sp.csr_matrix(output_shape, dtype=np.float64)
        return (empty, empty.copy(), [])
    degrees = _sparse_weighted_degrees_exact(graph)
    initial_values = degrees.sum() / num_nodes / degrees
    if spectral_gap_override is None:
        (laplacian, spectral_gap) = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)
    else:
        spectral_gap = float(spectral_gap_override)
        if not np.isfinite(spectral_gap) or spectral_gap <= 0:
            raise ValueError('spectral_gap_override must be finite and positive')
        (laplacian, _) = construct_laplacian(graph, use_spectral_gap=False)
        laplacian /= spectral_gap
    (peak_amplitudes, peak_pos, plan_metadata) = lazy_scipy_tiled_compute_selected_peaks_lazy_scipy_tiled(laplacian, initial_values, np.asarray(times), source_nodes, retain_nodes, tile_size=max(1, int(tile_size)), n_threads=min(32, max(1, int(n_threads))))
    initial_selected = np.zeros(output_shape, dtype=np.float64)
    retain_lookup = np.full(num_nodes, -1, dtype=np.int64)
    retain_lookup[retain_nodes] = np.arange(retain_nodes.size, dtype=np.int64)
    retained_source_rows = retain_lookup[source_nodes]
    present = retained_source_rows >= 0
    initial_selected[retained_source_rows[present], np.flatnonzero(present)] = initial_values[source_nodes[present]]
    peak_times = np.asarray(times)[peak_pos]
    unreachable = peak_amplitudes < 1.0 / num_nodes + PRECISION
    initial_nonzero = initial_selected > 0
    missed_peaks = (peak_pos == 0) & ~initial_nonzero
    if np.any(missed_peaks):
        warnings.warn('Some peaks are not detected because they occur at the minimum time step. Consider reducing the minimum time step in `times` to capture these peaks.')
    peak_times[unreachable] = np.nan
    peak_amplitudes[unreachable] = np.nan
    peak_times[initial_nonzero] = np.nan
    peak_amplitudes[initial_nonzero] = np.nan
    diffusion_coefficient = 0.5 / spectral_gap
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_dimensions = -2.0 * np.log(peak_amplitudes) / (1.0 + np.log(peak_times) + np.log(4.0 * diffusion_coefficient * np.pi))
    relative_dimensions[relative_dimensions < 0] = np.nan
    rd = sp.csr_matrix(np.nan_to_num(relative_dimensions, nan=0.0))
    pt = sp.csr_matrix(np.nan_to_num(peak_times, nan=0.0))
    rd.eliminate_zeros()
    pt.eliminate_zeros()
    rd.sort_indices()
    pt.sort_indices()
    logger.info('✅ Lazy SciPy-tiled selected strict RD processed: %d retained rows × %d sources.', retain_nodes.size, source_nodes.size)
    return (rd, pt, plan_metadata)

def _prepare_strict_expm_plans(laplacian, times):
    """Precompute source-independent SciPy expm parameters exactly."""
    if _scipy_expm is None:
        return None
    times = np.asarray(times)
    deltas = np.empty(times.shape, dtype=times.dtype)
    deltas[0] = times[0]
    deltas[1:] = times[1:] - times[:-1]
    plans = []
    tolerance = 2 ** (-53)
    n_nodes = laplacian.shape[0]
    try:
        for delta_t in deltas:
            matrix = -delta_t * laplacian
            identity = _scipy_expm._ident_like(matrix)
            mu = _scipy_expm._trace(matrix) / float(n_nodes)
            shifted = (matrix - mu * identity).tocsr()
            matrix_norm = _scipy_expm._exact_1_norm(shifted)
            if matrix_norm == 0:
                (degree, scaling) = (0, 1)
            else:
                norm_info = _scipy_expm.LazyOperatorNormInfo(shifted, A_1_norm=matrix_norm, ell=2)
                (degree, scaling) = _scipy_expm._fragment_3_1(norm_info, 1, tolerance, ell=2)
            plans.append((shifted, mu, degree, scaling))
    except (AttributeError, TypeError):
        logger.warning('SciPy expm internals are incompatible; using the uncached strict RD path.')
        return None
    return plans

def _pack_strict_expm_plans(plans):
    offsets = np.zeros(len(plans) + 1, dtype=np.int64)
    for (i, (matrix, _, _, _)) in enumerate(plans):
        offsets[i + 1] = offsets[i] + matrix.nnz
    data = np.concatenate([plan[0].data for plan in plans])
    indices = np.concatenate([plan[0].indices for plan in plans])
    indptr = np.stack([plan[0].indptr for plan in plans])
    mu = np.asarray([plan[1] for plan in plans], dtype=np.float64)
    degree = np.asarray([plan[2] for plan in plans], dtype=np.int64)
    scaling = np.asarray([plan[3] for plan in plans], dtype=np.int64)
    return (data, indices, indptr, offsets, mu, degree, scaling, plans[0][0].shape)

def _materialize_strict_expm_plan(matrix, mu, degree, scaling):
    degree = int(degree)
    scaling = int(scaling)
    eta = np.exp(mu / float(scaling))
    coefficients = np.asarray([1.0 / float(scaling * (j + 1)) for j in range(degree)], dtype=np.float64)
    return (matrix, eta, coefficients, scaling)

def _initialize_strict_rd_worker(packed_plans, initial_values, times, spectral_gap):
    global _STRICT_RD_WORKER_STATE
    (data, indices, indptr, offsets, mu, degree, scaling, shape) = packed_plans
    plans = []
    for i in range(len(mu)):
        start = offsets[i]
        end = offsets[i + 1]
        matrix = sp.csr_matrix((data[start:end], indices[start:end], indptr[i]), shape=shape, copy=False)
        plans.append(_materialize_strict_expm_plan(matrix, mu[i], degree[i], scaling[i]))
    _STRICT_RD_WORKER_STATE = (plans, initial_values, np.asarray(times), spectral_gap)

def _initialize_strict_rd_worker_from_file(plan_path):
    (packed_plans, initial_values, times, spectral_gap) = load(plan_path, mmap_mode='r')
    _initialize_strict_rd_worker(packed_plans, initial_values, times, spectral_gap)

def _apply_strict_expm_plan(plan, measure):
    (matrix, eta, coefficients, scaling) = plan
    current = measure
    result = measure
    tolerance = 2 ** (-53)
    for _ in range(scaling):
        previous_norm = _scipy_expm._exact_inf_norm(current)
        for coefficient in coefficients:
            current = coefficient * matrix.dot(current)
            current_norm = _scipy_expm._exact_inf_norm(current)
            result = result + current
            if previous_norm + current_norm <= tolerance * _scipy_expm._exact_inf_norm(result):
                break
            previous_norm = current_norm
        result = eta * result
        current = result
    return result

def _compute_node_trajectory_peaks_from_plans(plans, initial_measure):
    current = _apply_strict_expm_plan(plans[0], initial_measure)
    peak_amplitudes = current.copy()
    peak_pos = np.zeros(current.shape, dtype=np.int64)
    for (i, plan) in enumerate(plans[1:], start=1):
        current = _apply_strict_expm_plan(plan, current)
        update = current > peak_amplitudes
        peak_amplitudes[update] = current[update]
        peak_pos[update] = i
    return (peak_amplitudes, peak_pos)

def _validate_strict_expm_plans(laplacian, initial_values, times, plans):
    source = np.zeros(initial_values.size, dtype=np.float64)
    source[0] = initial_values[0]
    reference = source
    candidate = source
    materialized = [_materialize_strict_expm_plan(*plan) for plan in plans]
    for (i, plan) in enumerate(materialized):
        timestep = times[0] if i == 0 else times[i] - times[i - 1]
        reference = heat_kernel(laplacian, timestep, reference)
        candidate = _apply_strict_expm_plan(plan, candidate)
        if not np.array_equal(reference, candidate):
            logger.warning('Cached SciPy expm plan differs from the public strict path; using the uncached implementation.')
            return False
    return True

def _process_cached_strict_source(source_idx):
    (plans, initial_values, times, spectral_gap) = _STRICT_RD_WORKER_STATE
    len_graph = initial_values.size
    source = np.zeros(len_graph, dtype=np.float64)
    source[source_idx] = initial_values[source_idx]
    (peak_amplitudes, peak_pos) = _compute_node_trajectory_peaks_from_plans(plans, source)
    (rd_batch, pt_batch, _, _) = _extract_relative_dimensions_from_peaks(times, peak_amplitudes[:, None], peak_pos[:, None], source[:, None], spectral_gap)
    rd_source = rd_batch[:, 0]
    pt_source = pt_batch[:, 0]
    rd_rows = np.flatnonzero(~np.isnan(rd_source)).astype(np.int32)
    pt_rows = np.flatnonzero(~np.isnan(pt_source)).astype(np.int32)
    return (rd_rows, rd_source[rd_rows], pt_rows, pt_source[pt_rows], source_idx)

def _process_cached_strict_source_batch(source_indices):
    return [_process_cached_strict_source(source_idx) for source_idx in source_indices]

def _strict_source_batches(num_sources, n_workers, source_block_size=1):
    target_batches = max(1, n_workers * STRICT_RD_BATCHES_PER_WORKER)
    batch_size = max(1, (num_sources + target_batches - 1) // target_batches)
    if source_block_size > 1:
        batch_size = (batch_size + source_block_size - 1) // source_block_size * source_block_size
    return [range(start, min(start + batch_size, num_sources)) for start in range(0, num_sources, batch_size)]

def _run_cached_strict_sources(plans, initial_values, times, spectral_gap, relative_dimensions, peak_times, n_workers, disable_tqdm, strict_block_size=1, strict_block_backend='scipy'):
    packed_plans = _pack_strict_expm_plans(plans)
    initializer_args = (packed_plans, initial_values, times, spectral_gap)
    if strict_block_size > 1:
        initializer_args += (None, strict_block_size, strict_block_backend)
    source_indices = range(initial_values.size)
    if n_workers == 1 and strict_block_size > 1:
        _strict_rd_worker.worker_initialize(*initializer_args)
        for batch in _strict_source_batches(initial_values.size, 1, strict_block_size):
            (sources, rp, rr, rv, pp, pr, pv) = _strict_rd_worker.worker_process_source_batch_packed(batch)
            relative_dimensions[rr, np.repeat(sources, np.diff(rp))] = rv
            peak_times[pr, np.repeat(sources, np.diff(pp))] = pv
    elif n_workers == 1:
        _initialize_strict_rd_worker(*initializer_args)
        results = (_process_cached_strict_source(source_idx) for source_idx in tqdm(source_indices, disable=disable_tqdm))
        for (rd_rows, rd_values, pt_rows, pt_values, source_idx) in results:
            relative_dimensions[rd_rows, source_idx] = rd_values
            peak_times[pt_rows, source_idx] = pt_values
    else:
        temp_directory = tempfile.TemporaryDirectory(prefix='cellorigin-rd-plans-')
        _STRICT_RD_TEMP_DIRECTORIES.append(temp_directory)
        plan_path = os.path.join(temp_directory.name, 'strict_expm_plans.joblib')
        dump(initializer_args, plan_path, compress=0)
        source_batches = _strict_source_batches(initial_values.size, n_workers, strict_block_size)
        results = Parallel(n_jobs=n_workers, prefer='processes', return_as='generator_unordered', batch_size='auto', initializer=_strict_rd_worker.worker_initialize_from_file, initargs=(plan_path,))((delayed(_strict_rd_worker.worker_process_source_batch_packed)(source_batch) for source_batch in tqdm(source_batches, disable=disable_tqdm)))
        for (source_batch, rd_indptr, rd_rows, rd_values, pt_indptr, pt_rows, pt_values) in results:
            rd_columns = np.repeat(source_batch, np.diff(rd_indptr))
            pt_columns = np.repeat(source_batch, np.diff(pt_indptr))
            relative_dimensions[rd_rows, rd_columns] = rd_values
            peak_times[pt_rows, pt_columns] = pt_values

def process_batch(args):
    """
    Process a batch of sources.

    Parameters:
    - args: Tuple containing (laplacian, sources_batch, times, spectral_gap, start_idx)

    Returns:
    - rd_batch: Relative dimensions for the batch
    - pt_batch: Peak times for the batch
    - start_idx: Start index of the batch
    """
    if len(args) == 5:
        (laplacian, sources_batch, times, spectral_gap, start_idx) = args
    else:
        (laplacian, initial_values, times, spectral_gap, start_idx, end_idx, len_graph) = args
        if end_idx == start_idx + 1:
            return process_single_source_batch(laplacian, initial_values, times, spectral_gap, start_idx, len_graph)
        sources_batch = _make_sources_batch(len_graph, initial_values, start_idx, end_idx)
    (peak_amplitudes, peak_pos) = compute_node_trajectory_peaks(laplacian, sources_batch, times)
    (rd_batch, pt_batch, _, _) = _extract_relative_dimensions_from_peaks(times, peak_amplitudes, peak_pos, sources_batch, spectral_gap)
    return (rd_batch, pt_batch, start_idx)

def _make_sources_batch(len_graph, initial_values, start_idx, end_idx):
    sources_batch = np.zeros((len_graph, end_idx - start_idx), dtype=np.float64)
    rows = np.arange(start_idx, end_idx)
    cols = np.arange(end_idx - start_idx)
    sources_batch[rows, cols] = initial_values[start_idx:end_idx]
    return sources_batch

def process_single_source_batch(laplacian, initial_values, times, spectral_gap, source_idx, len_graph):
    source = np.zeros(len_graph, dtype=np.float64)
    source[source_idx] = initial_values[source_idx]
    (peak_amplitudes, peak_pos) = compute_node_trajectory_peaks(laplacian, source, times)
    (rd_batch, pt_batch, _, _) = _extract_relative_dimensions_from_peaks(times, peak_amplitudes[:, None], peak_pos[:, None], source[:, None], spectral_gap)
    return (rd_batch, pt_batch, source_idx)

def construct_laplacian(graph, laplacian_type='normalized', use_spectral_gap=True):
    """
    Construct the graph Laplacian matrix with stable spectral gap handling.

    Parameters:
    - graph: NetworkX graph
    - laplacian_type: Type of Laplacian ("normalized" is currently supported)
    - use_spectral_gap: Whether to scale by the spectral gap

    Returns:
    - laplacian: Sparse Laplacian matrix
    - spectral_gap: Spectral gap of the Laplacian
    """
    if laplacian_type == 'normalized' and sp.issparse(graph):
        adjacency = sp.csr_array(graph)
        degrees = _sparse_weighted_degrees_exact(adjacency)
        if np.any(degrees == 0):
            raise ValueError('The graph contains zero-degree nodes.')
        (n_rows, n_cols) = adjacency.shape
        unnormalized = sp.csr_array(sp.spdiags(adjacency.sum(axis=1), 0, n_cols, n_rows, format='csr')) - adjacency
        laplacian = sp.diags(1.0 / degrees).dot(unnormalized)
    elif laplacian_type == 'normalized':
        degrees = np.array([_weighted_degree_exact(graph, i) for i in graph.nodes()])
        laplacian = sp.diags(1.0 / degrees).dot(nx.laplacian_matrix(graph))
    else:
        raise NotImplementedError("Only 'normalized' Laplacian is implemented.")
    spectral_gap = 1.0
    if use_spectral_gap:
        try:
            spectral_gap_values = np.array([abs(sp.linalg.eigs(laplacian, which='SM', k=2)[0][1]) for _ in range(10)])
            max_diff = max(spectral_gap_values) - min(spectral_gap_values)
            spectral_gap = spectral_gap_values[0]
            spectral_gap_max = spectral_gap_values.max()
            if spectral_gap_max < SGAP_THRESHOLD or max_diff > CNVG_THRESHOLD:
                reason = []
                if spectral_gap_max < SGAP_THRESHOLD:
                    reason.append(f'too small ({spectral_gap_max:.6e} < {SGAP_THRESHOLD})')
                if max_diff > CNVG_THRESHOLD:
                    reason.append(f'unstable (max diff: {max_diff:.6e} > {CNVG_THRESHOLD})')
                reason_str = ' and '.join(reason)
                logger.warning('⚠️ NOTICE: Spectral gap is %s. Setting to 1.', reason_str)
                spectral_gap = 1.0
        except Exception as e:
            logger.exception('⚠️ eigs() failed. Setting spectral gap to 1.')
            spectral_gap = 1.0
        logger.info('Parameter spectral_gap set to %.6e', spectral_gap)
        laplacian /= spectral_gap
    return (laplacian, spectral_gap)

def compute_node_trajectories(laplacian, initial_measure, times, disable_tqdm=False):
    """
    Compute diffusion trajectories for nodes.

    Parameters:
    - laplacian: Laplacian matrix
    - initial_measure: Initial measure for diffusion
    - times: List of time steps

    Returns:
    - node_trajectories: Array of trajectories over time
    """
    node_trajectories = [heat_kernel(laplacian, times[0], initial_measure)]
    for i in range(1, len(times)):
        delta_t = times[i] - times[i - 1]
        node_trajectories.append(heat_kernel(laplacian, delta_t, node_trajectories[-1]))
    return np.array(node_trajectories)

def compute_node_trajectory_peaks(laplacian, initial_measure, times):
    """
    Compute diffusion peak amplitudes and positions without storing all times.
    """
    current = heat_kernel(laplacian, times[0], initial_measure)
    peak_amplitudes = current.copy()
    peak_pos = np.zeros(current.shape, dtype=np.int64)
    for i in range(1, len(times)):
        delta_t = times[i] - times[i - 1]
        current = heat_kernel(laplacian, delta_t, current)
        update = current > peak_amplitudes
        peak_amplitudes[update] = current[update]
        peak_pos[update] = i
    return (peak_amplitudes, peak_pos)

def heat_kernel(laplacian, timestep, measure):
    """
    Compute the heat kernel for diffusion.

    Parameters:
    - laplacian: Laplacian matrix
    - timestep: Time step for diffusion
    - measure: Initial measure

    Returns:
    - Result of matrix exponential applied to the measure
    """
    return sp.linalg.expm_multiply(-timestep * laplacian, measure)

def extract_relative_dimensions_vectorized(times, node_trajectories, initial_measures, spectral_gap):
    """
    Compute relative dimensions from node trajectories, vectorized over sources.

    Parameters:
    - times: List of time steps
    - node_trajectories: Diffusion trajectories (n_times, n_nodes, n_sources)
    - initial_measures: Initial measures for each node (n_nodes, n_sources)
    - spectral_gap: Spectral gap for normalization

    Returns:
    - relative_dimensions: Array of relative dimensions
    - peak_times: Array of peak times
    - peak_amplitudes: Array of peak amplitudes
    - diffusion_coefficient: Diffusion coefficient
    """
    peak_amplitudes = np.max(node_trajectories, axis=0)
    peak_pos = np.argmax(node_trajectories, axis=0)
    return _extract_relative_dimensions_from_peaks(times, peak_amplitudes, peak_pos, initial_measures, spectral_gap)

def _extract_relative_dimensions_from_peaks(times, peak_amplitudes, peak_pos, initial_measures, spectral_gap):
    n_nodes = peak_amplitudes.shape[0]
    stationary_prob = 1 / n_nodes
    peak_times = times[peak_pos]
    unreachable = peak_amplitudes < stationary_prob + PRECISION
    initial_measures_nonzero = initial_measures > 0
    missed_peaks = (peak_pos == 0) & ~initial_measures_nonzero
    if np.any(missed_peaks):
        warnings.warn('Some peaks are not detected because they occur at the minimum time step. Consider reducing the minimum time step in `times` to capture these peaks.')
    peak_times[unreachable] = np.nan
    peak_amplitudes[unreachable] = np.nan
    peak_times[initial_measures_nonzero] = np.nan
    peak_amplitudes[initial_measures_nonzero] = np.nan
    diffusion_coefficient = 0.5 / spectral_gap
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_dimensions = -2.0 * np.log(peak_amplitudes) / (1.0 + np.log(peak_times) + np.log(4.0 * diffusion_coefficient * np.pi))
    relative_dimensions[relative_dimensions < 0] = np.nan
    return (relative_dimensions, peak_times, peak_amplitudes, diffusion_coefficient)

def run_single_source(graph, times, source_node, use_spectral_gap=True):
    """
    Compute relative dimensions for a single source node in a graph.

    Parameters:
    - graph: NetworkX graph
    - times: List of time steps for diffusion
    - source_node: Node name in the graph to act as the source
    - use_spectral_gap: Whether to normalize Laplacian by spectral gap

    Returns:
    - results: Dictionary containing:
        - "relative_dimensions": Array of relative dimensions for all target nodes
        - "peak_amplitudes": Array of peak amplitudes
        - "peak_times": Array of peak times
        - "diffusion_coefficient": Diffusion coefficient used
        - "times": List of time steps
        - "node_trajectories": Array of diffusion trajectories
    """
    (laplacian, spectral_gap) = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)
    initial_measure = get_initial_measure(graph, [source_node])
    node_trajectories = compute_node_trajectories(laplacian, initial_measure, times)
    (relative_dimensions, peak_times, peak_amplitudes, diffusion_coefficient) = extract_relative_dimensions_vectorized(times, node_trajectories, initial_measure[:, None], spectral_gap)
    results = {'relative_dimensions': relative_dimensions.flatten(), 'peak_amplitudes': peak_amplitudes.flatten(), 'peak_times': peak_times.flatten(), 'diffusion_coefficient': diffusion_coefficient, 'times': times, 'node_trajectories': node_trajectories}
    return results

def get_initial_measure(graph, nodes):
    """
    Create a measure with the correct mass distributed among the given nodes.

    Parameters:
    - graph: NetworkX graph
    - nodes: List of nodes to distribute the measure

    Returns:
    - measure: 1D numpy array of size n_nodes, representing the initial measure
    """
    measure = np.zeros(len(graph), dtype=np.float64)
    if len(graph) == 0:
        raise ValueError('The graph is empty. Cannot compute initial measure.')
    if not nodes:
        raise ValueError('The nodes list is empty. Provide at least one node.')
    total_degree = _ordered_sum((_weighted_degree_exact(graph, u) for u in graph))
    if total_degree == 0:
        raise ValueError('The graph has zero total degree. Check for disconnected nodes or weights.')
    for node in nodes:
        degree = _weighted_degree_exact(graph, node)
        if degree == 0:
            raise ValueError(f'Node {node} has zero degree and cannot contribute to the measure.')
        measure[node] = total_degree / (len(graph) * degree * len(nodes))
    return measure

@calc_time
def calculate_relative_dimension(adata: AnnData, graph: nx.Graph, time_arr: np.array, use_spectral_gap=True, reld_key: str='relative_dimension', batch_size: int=1, num_processes: int=None, strict_block_size: int=1, strict_block_backend: str='scipy'):
    """
    Calculate the relative dimension matrix and automatically add it to the AnnData object.

    Parameters:
    - adata: The input AnnData object.
    - time_arr: Array of time steps for diffusion.
    - obsp_key: Key to the graph in obsp.
    - reld_key: Key to store the relative dimension matrix in obsp.
    - batch_size: Number of nodes to process in each batch.
    - num_processes: Number of workers for parallel processing. If `None`, use all available CPUs.
    - strict_block_size: Opt-in source-major strict blocks; requires batch_size=1.
    - strict_block_backend: 'scipy' or opt-in buffered 'numba' block products.
    """
    rdim = run_all_sources(graph, time_arr, batch_size=batch_size, n_workers=num_processes, use_spectral_gap=use_spectral_gap, strict_block_size=strict_block_size, strict_block_backend=strict_block_backend)[0]
    nanmin = np.nanmin(rdim)
    if nanmin != 0:
        logger.info('✅ Minimum non-NaN value in the relative dimension matrix is %.6e. Filling NaN values with 0 for sparse representation.', nanmin)
        rdim[np.isnan(rdim)] = 0
        rdim_sparse = sp.csr_matrix(rdim)
        adata.obsp[reld_key] = rdim_sparse
        logger.info("✅ Relative dimension matrix with key '%s' added to `adata.obsp` as a sparse matrix.", reld_key)
    else:
        adata.obsp[reld_key] = rdim
        logger.info("✅ Relative dimension matrix with key '%s' added to `adata.obsp`.", reld_key)
    return None
