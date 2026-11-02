"""Internal exact-RD kernels and worker utilities."""
from __future__ import annotations
'Utilities: worker'
import warnings
import numpy as np
import scipy.sparse as sp
from joblib import load
try:
    from scipy.sparse.linalg import _expm_multiply as _scipy_expm
except ImportError:
    _scipy_expm = None
worker_PRECISION = 1e-08
worker__STATE = None
worker__SOURCE_BLOCK_SIZE = 1
worker__SOURCE_BLOCK_BACKEND = 'scipy'

def worker__materialize_plan(matrix, mu, degree, scaling):
    degree = int(degree)
    scaling = int(scaling)
    eta = np.exp(mu / float(scaling))
    coefficients = np.asarray([1.0 / float(scaling * (j + 1)) for j in range(degree)], dtype=np.float64)
    return (matrix, eta, coefficients, scaling)

def worker_initialize(packed_plans, initial_values, times, spectral_gap, output_row_lookup=None, source_block_size=1, source_block_backend='scipy'):
    global worker__STATE, worker__SOURCE_BLOCK_SIZE, worker__SOURCE_BLOCK_BACKEND
    worker__SOURCE_BLOCK_SIZE = source_block_size
    worker__SOURCE_BLOCK_BACKEND = source_block_backend
    (data, indices, indptr, offsets, mu, degree, scaling, shape) = packed_plans
    plans = []
    for i in range(len(mu)):
        start = offsets[i]
        end = offsets[i + 1]
        matrix = sp.csr_matrix((data[start:end], indices[start:end], indptr[i]), shape=shape, copy=False)
        plans.append(worker__materialize_plan(matrix, mu[i], degree[i], scaling[i]))
    worker__STATE = (plans, initial_values, np.asarray(times), spectral_gap, output_row_lookup)

def worker_initialize_from_file(plan_path):
    worker_initialize(*load(plan_path, mmap_mode='r'))

def worker__apply_plan(plan, measure):
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

def worker__compute_peaks(plans, initial_measure):
    current = worker__apply_plan(plans[0], initial_measure)
    peak_amplitudes = current.copy()
    peak_pos = np.zeros(current.shape, dtype=np.int64)
    for (i, plan) in enumerate(plans[1:], start=1):
        current = worker__apply_plan(plan, current)
        update = current > peak_amplitudes
        peak_amplitudes[update] = current[update]
        peak_pos[update] = i
    return (peak_amplitudes, peak_pos)

def worker__extract_relative_dimensions(times, peak_amplitudes, peak_pos, initial_measures, spectral_gap):
    n_nodes = peak_amplitudes.shape[0]
    stationary_prob = 1 / n_nodes
    peak_times = times[peak_pos]
    unreachable = peak_amplitudes < stationary_prob + worker_PRECISION
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
    return (relative_dimensions, peak_times)

def worker_process_source(source_idx):
    (plans, initial_values, times, spectral_gap, output_row_lookup) = worker__STATE
    len_graph = initial_values.size
    source = np.zeros(len_graph, dtype=np.float64)
    source[source_idx] = initial_values[source_idx]
    (peak_amplitudes, peak_pos) = worker__compute_peaks(plans, source)
    (rd_batch, pt_batch) = worker__extract_relative_dimensions(times, peak_amplitudes[:, None], peak_pos[:, None], source[:, None], spectral_gap)
    rd_source = rd_batch[:, 0]
    pt_source = pt_batch[:, 0]
    rd_rows = np.flatnonzero(~np.isnan(rd_source)).astype(np.int32)
    pt_rows = np.flatnonzero(~np.isnan(pt_source)).astype(np.int32)
    if output_row_lookup is not None:
        rd_local_rows = output_row_lookup[rd_rows]
        pt_local_rows = output_row_lookup[pt_rows]
        rd_keep = rd_local_rows >= 0
        pt_keep = pt_local_rows >= 0
        rd_rows = rd_local_rows[rd_keep].astype(np.int32, copy=False)
        pt_rows = pt_local_rows[pt_keep].astype(np.int32, copy=False)
        rd_values = rd_source[np.flatnonzero(~np.isnan(rd_source))[rd_keep]]
        pt_values = pt_source[np.flatnonzero(~np.isnan(pt_source))[pt_keep]]
    else:
        rd_values = rd_source[rd_rows]
        pt_values = pt_source[pt_rows]
    return (rd_rows, rd_values, pt_rows, pt_values, source_idx)

def worker_process_source_batch(source_indices):
    return [worker_process_source(source_idx) for source_idx in source_indices]

def worker_process_source_batch_packed(source_indices):
    if worker__SOURCE_BLOCK_SIZE > 1:
        results = block_process_source_batch(source_indices, worker__SOURCE_BLOCK_SIZE, worker__STATE, worker__SOURCE_BLOCK_BACKEND)
    else:
        results = worker_process_source_batch(source_indices)
    sources = np.asarray([result[4] for result in results], dtype=np.int64)
    rd_counts = np.asarray([result[0].size for result in results], dtype=np.int64)
    pt_counts = np.asarray([result[2].size for result in results], dtype=np.int64)
    rd_indptr = np.empty(sources.size + 1, dtype=np.int64)
    pt_indptr = np.empty(sources.size + 1, dtype=np.int64)
    rd_indptr[0] = 0
    pt_indptr[0] = 0
    np.cumsum(rd_counts, out=rd_indptr[1:])
    np.cumsum(pt_counts, out=pt_indptr[1:])
    return (sources, rd_indptr, np.concatenate([result[0] for result in results]), np.concatenate([result[1] for result in results]), pt_indptr, np.concatenate([result[2] for result in results]), np.concatenate([result[3] for result in results]))
'Utilities: block'
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def block__column_norms(values):
    norms = np.zeros(values.shape[1])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = abs(values[i, j])
            if value > norms[j] or np.isnan(value):
                norms[j] = value
    return norms

@njit(cache=True, fastmath=False)
def block__accumulate_active(term, result, coefficient, active, previous_norms):
    norms = np.zeros(term.shape[1])
    result_norms = np.zeros(term.shape[1])
    for i in range(term.shape[0]):
        for j in range(term.shape[1]):
            if active[j]:
                value = coefficient * term[i, j]
                total = result[i, j] + value
                term[i, j] = value
                result[i, j] = total
                (a, b) = (abs(value), abs(total))
                if a > norms[j] or np.isnan(a):
                    norms[j] = a
                if b > result_norms[j] or np.isnan(b):
                    result_norms[j] = b
    remaining = 0
    for j in range(term.shape[1]):
        if active[j]:
            if previous_norms[j] + norms[j] <= 2.0 ** (-53) * result_norms[j]:
                active[j] = False
            else:
                remaining += 1
            previous_norms[j] = norms[j]
    return remaining

@njit(cache=True, fastmath=False)
def block__update_peaks(current, amplitudes, positions, time_index):
    for i in range(current.size):
        if current[i] > amplitudes[i]:
            amplitudes[i] = current[i]
            positions[i] = time_index

def block_apply_plan(plan, measure):
    """Reuse scalar expm plans, never a multi-RHS stopping criterion."""
    (matrix, eta, coefficients, scaling) = plan
    current = measure
    result = measure.copy(order='C')
    for _ in range(scaling):
        previous_norms = block__column_norms(current)
        active = np.ones(current.shape[1], dtype=np.bool_)
        for coefficient in coefficients:
            current = matrix.dot(current)
            if not block__accumulate_active(current, result, coefficient, active, previous_norms):
                break
        np.multiply(result, eta, out=result)
        current = result
    return result

def block__plan_function(backend):
    if backend == 'scipy':
        return block_apply_plan
    if backend == 'numba':
        return compiled_apply_plan
    if backend == 'register':
        return register_apply
    raise ValueError("strict_block_backend must be 'scipy', 'numba', or 'register'")

def block_compute_peaks(plans, initial_measure, backend='scipy'):
    apply = block__plan_function(backend)
    current = apply(plans[0], initial_measure)
    amplitudes = current.copy()
    positions = np.zeros(current.shape, dtype=np.int64)
    for (i, plan) in enumerate(plans[1:], start=1):
        current = apply(plan, current)
        block__update_peaks(current.ravel(), amplitudes.ravel(), positions.ravel(), i)
    return (amplitudes, positions)

def block_validate_plans(plans, initial_values, block_size, backend='scipy'):
    """Reject the block path if sampled trajectories differ on this runtime."""
    apply = block__plan_function(backend)
    width = min(block_size, initial_values.size)
    sources = np.linspace(0, initial_values.size - 1, width, dtype=int)
    reference = np.zeros((initial_values.size, width))
    reference[sources, np.arange(width)] = initial_values[sources]
    candidate = reference.copy()
    for plan in plans:
        reference = np.column_stack([worker__apply_plan(plan, reference[:, j]) for j in range(width)])
        candidate = apply(plan, candidate)
        if not np.array_equal(reference, candidate):
            return False
    return True

def block_process_source_batch(source_indices, block_size, state, backend='scipy'):
    (plans, initial, times, gap, lookup) = state
    if lookup is not None:
        raise ValueError('Source blocks currently require all retained rows')
    source_indices = list(source_indices)
    results = []
    for start in range(0, len(source_indices), block_size):
        ids = source_indices[start:start + block_size]
        source = np.zeros((initial.size, len(ids)))
        source[ids, np.arange(len(ids))] = initial[ids]
        (amplitudes, positions) = block_compute_peaks(plans, source, backend)
        (rd, pt) = worker__extract_relative_dimensions(times, amplitudes, positions, source, gap)
        for (j, source_id) in enumerate(ids):
            rows = np.flatnonzero(~np.isnan(rd[:, j])).astype(np.int32)
            pt_rows = np.flatnonzero(~np.isnan(pt[:, j])).astype(np.int32)
            results.append((rows, rd[rows, j], pt_rows, pt[pt_rows, j], source_id))
    return results
'Utilities: compiled'
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def compiled__csr_product(data, indices, indptr, current, target):
    for i in range(current.shape[0]):
        for j in range(current.shape[1]):
            target[i, j] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            (value, k) = (data[p], indices[p])
            for j in range(current.shape[1]):
                target[i, j] += value * current[k, j]

@njit(cache=True, fastmath=False)
def compiled__apply_plan(data, indices, indptr, eta, coefficients, scaling, measure):
    result = measure.copy()
    current = np.empty_like(measure)
    target = np.empty_like(measure)
    for _ in range(scaling):
        current[:, :] = result
        previous = block__column_norms(current)
        active = np.ones(current.shape[1], dtype=np.bool_)
        for coefficient in coefficients:
            compiled__csr_product(data, indices, indptr, current, target)
            remaining = block__accumulate_active(target, result, coefficient, active, previous)
            (current, target) = (target, current)
            if not remaining:
                break
        result *= eta
    return result

def compiled_apply_plan(plan, measure):
    (matrix, eta, coefficients, scaling) = plan
    return compiled__apply_plan(matrix.data, matrix.indices, matrix.indptr, eta, coefficients, scaling, np.ascontiguousarray(measure))
'Utilities: register'
import ctypes
import json
import os
from pathlib import Path
import numpy as np
from sysconfig import get_config_var
register_lib = ctypes.CDLL(str(Path(__file__).parent / '_native' / ('register_product' + get_config_var('EXT_SUFFIX'))))
register_product = register_lib.csr_product16
register_ptr = np.ctypeslib.ndpointer
register_product.argtypes = [ctypes.c_int64, register_ptr(np.float64, ndim=1, flags='C_CONTIGUOUS'), register_ptr(np.int32, ndim=1, flags='C_CONTIGUOUS'), register_ptr(np.int32, ndim=1, flags='C_CONTIGUOUS'), register_ptr(np.float64, ndim=2, flags='C_CONTIGUOUS'), register_ptr(np.float64, ndim=2, flags='C_CONTIGUOUS')]
register_product.restype = None
register__seen_paths = set()

def register__record_path(mode, width):
    """Optional first-use evidence, never a per-product filesystem operation."""
    destination = os.environ.get('CELLORIGIN_REGISTER_AUDIT')
    key = (mode, width)
    if destination and key not in register__seen_paths:
        record = dict(pid=os.getpid(), mode=mode, width=width, module=__file__)
        path = Path(destination) / f'{os.getpid()}_{mode}_{width}.json'
        with path.open('x') as handle:
            json.dump(record, handle)
        register__seen_paths.add(key)

def register_apply(plan, measure):
    (matrix, eta, coefficients, scaling) = plan
    if measure.shape[1] != 16 or matrix.indices.dtype != np.int32 or matrix.indptr.dtype != np.int32:
        register__record_path('numba_fallback', measure.shape[1])
        return compiled_apply_plan(plan, measure)
    register__record_path('native', measure.shape[1])
    result = measure.copy(order='C')
    current = np.empty_like(result)
    target = np.empty_like(result)
    for _ in range(scaling):
        current[:, :] = result
        previous = block__column_norms(current)
        active = np.ones(16, dtype=np.bool_)
        for coefficient in coefficients:
            register_product(current.shape[0], matrix.data, matrix.indices, matrix.indptr, current, target)
            remaining = block__accumulate_active(target, result, coefficient, active, previous)
            (current, target) = (target, current)
            if not remaining:
                break
        result *= eta
    return result
'Utilities: lazy_tiled'
import logging
import time
import numpy as np
from numba import njit, prange, set_num_threads
try:
    from scipy.sparse.linalg import _expm_multiply as _scipy_expm
except ImportError:
    _scipy_expm = None
lazy_tiled_logger = logging.getLogger(__name__)

@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def lazy_tiled__advance_plan_tiled(matrix_data, matrix_indices, matrix_indptr, eta, coefficients, scaling, current, tile_size):
    """Apply one cached SciPy expm plan to tiled source columns."""
    (n_nodes, n_sources) = current.shape
    n_tiles = (n_sources + tile_size - 1) // tile_size
    tolerance = 2.0 ** (-53)
    for tile_idx in prange(n_tiles):
        source_start = tile_idx * tile_size
        source_end = min(source_start + tile_size, n_sources)
        width = source_end - source_start
        tile_current = current[:, source_start:source_end].copy()
        result = tile_current.copy()
        term = tile_current.copy()
        scratch = np.empty((n_nodes, width), dtype=np.float64)
        for _ in range(scaling):
            previous_norm = np.zeros(width, dtype=np.float64)
            active = np.ones(width, dtype=np.bool_)
            for row in range(n_nodes):
                for local_source in range(width):
                    value = abs(term[row, local_source])
                    if value > previous_norm[local_source]:
                        previous_norm[local_source] = value
            for coefficient in coefficients:
                for row in range(n_nodes):
                    for local_source in range(width):
                        scratch[row, local_source] = 0.0
                    edge_start = matrix_indptr[row]
                    edge_end = matrix_indptr[row + 1]
                    for edge_idx in range(edge_start, edge_end):
                        matrix_value = matrix_data[edge_idx]
                        column = matrix_indices[edge_idx]
                        for local_source in range(width):
                            if active[local_source]:
                                scratch[row, local_source] += matrix_value * term[column, local_source]
                    for local_source in range(width):
                        if active[local_source]:
                            scratch[row, local_source] = coefficient * scratch[row, local_source]
                current_norm = np.zeros(width, dtype=np.float64)
                result_norm = np.zeros(width, dtype=np.float64)
                for row in range(n_nodes):
                    for local_source in range(width):
                        if active[local_source]:
                            value = abs(scratch[row, local_source])
                            if value > current_norm[local_source]:
                                current_norm[local_source] = value
                            result[row, local_source] = result[row, local_source] + scratch[row, local_source]
                            value = abs(result[row, local_source])
                            if value > result_norm[local_source]:
                                result_norm[local_source] = value
                (term, scratch) = (scratch, term)
                any_active = False
                for local_source in range(width):
                    if active[local_source]:
                        if previous_norm[local_source] + current_norm[local_source] <= tolerance * result_norm[local_source]:
                            active[local_source] = False
                        else:
                            previous_norm[local_source] = current_norm[local_source]
                            any_active = True
                if not any_active:
                    break
            for row in range(n_nodes):
                for local_source in range(width):
                    result[row, local_source] = eta * result[row, local_source]
                    term[row, local_source] = result[row, local_source]
        current[:, source_start:source_end] = result

def lazy_tiled__prepare_single_expm_plan(laplacian, delta_t):
    """Reproduce one element of rdim._prepare_strict_expm_plans exactly."""
    if _scipy_expm is None:
        raise RuntimeError('SciPy expm internals are unavailable')
    matrix = -delta_t * laplacian
    identity = _scipy_expm._ident_like(matrix)
    mu = _scipy_expm._trace(matrix) / float(laplacian.shape[0])
    shifted = (matrix - mu * identity).tocsr()
    matrix_norm = _scipy_expm._exact_1_norm(shifted)
    if matrix_norm == 0:
        (degree, scaling) = (0, 1)
    else:
        norm_info = _scipy_expm.LazyOperatorNormInfo(shifted, A_1_norm=matrix_norm, ell=2)
        (degree, scaling) = _scipy_expm._fragment_3_1(norm_info, 1, 2 ** (-53), ell=2)
    eta = np.exp(mu / float(scaling))
    coefficients = np.asarray([1.0 / float(scaling * (index + 1)) for index in range(degree)], dtype=np.float64)
    return (shifted, eta, coefficients, int(scaling), int(degree), float(mu))

def lazy_tiled_compute_selected_peaks_lazy_tiled(laplacian, initial_values, times, source_nodes, retain_nodes, tile_size, n_threads):
    """Compute selected peaks while retaining only one expm plan at a time."""
    set_num_threads(max(1, int(n_threads)))
    times = np.asarray(times)
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    retain_nodes = np.asarray(retain_nodes, dtype=np.int64)
    tile_size = max(1, int(tile_size))
    current = np.zeros((initial_values.size, source_nodes.size), dtype=np.float64)
    current[source_nodes, np.arange(source_nodes.size)] = initial_values[source_nodes]
    peak_amplitudes = np.empty((retain_nodes.size, source_nodes.size), dtype=np.float64)
    peak_pos = np.zeros((retain_nodes.size, source_nodes.size), dtype=np.int64)
    deltas = np.empty(times.shape, dtype=times.dtype)
    deltas[0] = times[0]
    deltas[1:] = times[1:] - times[:-1]
    plan_metadata = []
    for (plan_index, delta_t) in enumerate(deltas):
        plan_started = time.perf_counter()
        (shifted, eta, coefficients, scaling, degree, mu) = lazy_tiled__prepare_single_expm_plan(laplacian, delta_t)
        lazy_tiled__advance_plan_tiled(shifted.data, shifted.indices, shifted.indptr, eta, coefficients, scaling, current, tile_size)
        selected = current[retain_nodes]
        if plan_index == 0:
            peak_amplitudes[:] = selected
        else:
            update = selected > peak_amplitudes
            peak_amplitudes[update] = selected[update]
            peak_pos[update] = plan_index
        plan_metadata.append({'degree': degree, 'scaling': scaling, 'mu': mu, 'nnz': int(shifted.nnz)})
        lazy_tiled_logger.info('Lazy strict RD plan %d/%d completed in %.2f seconds (degree=%d, scaling=%d).', plan_index + 1, deltas.size, time.perf_counter() - plan_started, degree, scaling)
    return (peak_amplitudes, peak_pos, plan_metadata)
'Utilities: lazy_threaded'
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import numpy as np
try:
    from scipy.sparse.linalg import _expm_multiply as _scipy_expm
except ImportError:
    _scipy_expm = None
lazy_threaded_logger = logging.getLogger(__name__)

def lazy_threaded__apply_prepared_plan(plan, measure):
    """Match rdim._apply_strict_expm_plan for one one-dimensional source."""
    if _scipy_expm is None:
        raise RuntimeError('SciPy expm internals are unavailable')
    (matrix, eta, coefficients, scaling) = plan
    current = measure
    result = measure
    tolerance = 2.0 ** (-53)
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

def lazy_threaded_compute_selected_peaks_lazy_threaded(laplacian, initial_values, times, source_nodes, retain_nodes, n_threads):
    """Compute exact selected peaks while retaining only one plan at a time."""
    times = np.asarray(times)
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    retain_nodes = np.asarray(retain_nodes, dtype=np.int64)
    n_threads = min(max(1, int(n_threads)), max(1, source_nodes.size))
    current = np.zeros((initial_values.size, source_nodes.size), dtype=np.float64, order='F')
    current[source_nodes, np.arange(source_nodes.size)] = initial_values[source_nodes]
    peak_amplitudes = np.empty((retain_nodes.size, source_nodes.size), dtype=np.float64)
    peak_pos = np.zeros((retain_nodes.size, source_nodes.size), dtype=np.int64)
    deltas = np.empty(times.shape, dtype=times.dtype)
    deltas[0] = times[0]
    deltas[1:] = times[1:] - times[:-1]
    plan_metadata = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for (plan_index, delta_t) in enumerate(deltas):
            plan_started = time.perf_counter()
            (shifted, eta, coefficients, scaling, degree, mu) = lazy_tiled__prepare_single_expm_plan(laplacian, delta_t)
            plan = (shifted, eta, coefficients, scaling)
            advanced = np.empty_like(current, order='F')
            results = executor.map(lambda values: lazy_threaded__apply_prepared_plan(plan, values), (current[:, index] for index in range(source_nodes.size)))
            for (source_index, values) in enumerate(results):
                advanced[:, source_index] = values
            current = advanced
            selected = current[retain_nodes]
            if plan_index == 0:
                peak_amplitudes[:] = selected
            else:
                update = selected > peak_amplitudes
                peak_amplitudes[update] = selected[update]
                peak_pos[update] = plan_index
            plan_metadata.append({'degree': degree, 'scaling': scaling, 'mu': mu, 'nnz': int(shifted.nnz)})
            lazy_threaded_logger.info('Lazy threaded strict RD plan %d/%d completed in %.2f seconds (degree=%d, scaling=%d).', plan_index + 1, deltas.size, time.perf_counter() - plan_started, degree, scaling)
    return (peak_amplitudes, peak_pos, plan_metadata)
'Utilities: lazy_scipy_tiled'
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import numpy as np
lazy_scipy_tiled_logger = logging.getLogger(__name__)

def lazy_scipy_tiled__column_inf_norm(values):
    return np.max(np.abs(values), axis=0)

def lazy_scipy_tiled__apply_prepared_plan_tile(plan, measure):
    """Apply one strict plan with independent per-column stopping decisions."""
    (matrix, eta, coefficients, scaling) = plan
    result = np.array(measure, dtype=np.float64, order='F', copy=True)
    term = result.copy(order='F')
    tolerance = 2.0 ** (-53)
    for _ in range(scaling):
        previous_norm = lazy_scipy_tiled__column_inf_norm(term)
        active = np.ones(term.shape[1], dtype=bool)
        for coefficient in coefficients:
            active_indices = np.flatnonzero(active)
            if not active_indices.size:
                break
            active_term = np.asfortranarray(term[:, active_indices])
            next_values = coefficient * matrix.dot(active_term)
            current_norm = lazy_scipy_tiled__column_inf_norm(next_values)
            result[:, active_indices] = result[:, active_indices] + next_values
            result_norm = lazy_scipy_tiled__column_inf_norm(result[:, active_indices])
            converged = previous_norm[active_indices] + current_norm <= tolerance * result_norm
            term[:, active_indices] = next_values
            previous_norm[active_indices] = current_norm
            active[active_indices[converged]] = False
        result *= eta
        term = result.copy(order='F')
    return result

def lazy_scipy_tiled_compute_selected_peaks_lazy_scipy_tiled(laplacian, initial_values, times, source_nodes, retain_nodes, tile_size, n_threads):
    """Compute exact selected peaks with threaded SciPy dense-column tiles."""
    times = np.asarray(times)
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    retain_nodes = np.asarray(retain_nodes, dtype=np.int64)
    tile_size = max(1, int(tile_size))
    tile_ranges = [(start, min(start + tile_size, source_nodes.size)) for start in range(0, source_nodes.size, tile_size)]
    n_threads = min(max(1, int(n_threads)), max(1, len(tile_ranges)))
    current = np.zeros((initial_values.size, source_nodes.size), dtype=np.float64, order='F')
    current[source_nodes, np.arange(source_nodes.size)] = initial_values[source_nodes]
    peak_amplitudes = np.empty((retain_nodes.size, source_nodes.size), dtype=np.float64)
    peak_pos = np.zeros((retain_nodes.size, source_nodes.size), dtype=np.int64)
    deltas = np.empty(times.shape, dtype=times.dtype)
    deltas[0] = times[0]
    deltas[1:] = times[1:] - times[:-1]
    plan_metadata = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for (plan_index, delta_t) in enumerate(deltas):
            plan_started = time.perf_counter()
            (shifted, eta, coefficients, scaling, degree, mu) = lazy_tiled__prepare_single_expm_plan(laplacian, delta_t)
            plan = (shifted, eta, coefficients, scaling)
            advanced = np.empty_like(current, order='F')
            results = executor.map(lambda bounds: lazy_scipy_tiled__apply_prepared_plan_tile(plan, current[:, bounds[0]:bounds[1]]), tile_ranges)
            for ((start, stop), values) in zip(tile_ranges, results):
                advanced[:, start:stop] = values
            current = advanced
            selected = current[retain_nodes]
            if plan_index == 0:
                peak_amplitudes[:] = selected
            else:
                update = selected > peak_amplitudes
                peak_amplitudes[update] = selected[update]
                peak_pos[update] = plan_index
            plan_metadata.append({'degree': degree, 'scaling': scaling, 'mu': mu, 'nnz': int(shifted.nnz)})
            lazy_scipy_tiled_logger.info('Lazy SciPy-tiled strict RD plan %d/%d completed in %.2f seconds (degree=%d, scaling=%d).', plan_index + 1, deltas.size, time.perf_counter() - plan_started, degree, scaling)
    return (peak_amplitudes, peak_pos, plan_metadata)
