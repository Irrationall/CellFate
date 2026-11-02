from multiprocessing import Pool, cpu_count
from anndata import AnnData
from cellfate._native import MINE
try:
    from cellfate._native import mic_pairs, mic_pstats, mic_pstats_filtered
except ImportError:
    mic_pairs = None
    mic_pstats = None
    mic_pstats_filtered = None
import scipy.sparse as sp
import scanpy as sc
import numpy as np
import time
from typing import Optional, Dict, Sequence, Tuple
from functools import wraps
from packaging import version
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
AUTO_MAX_THREADS = 128
AUTO_MAX_PROCESSES = 16
MIC_BLOCKS_PER_PROCESS = 64

def calc_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info("⏳ Function '%s' executed in %.2f seconds", func.__name__, end_time - start_time)
        return result
    return wrapper
_mic_data = None
_mic_candidate_rows = None
_mic_candidate_cols = None
mine = None

def init_mine(data=None):
    """Initializer for multiprocessing to avoid reinitializing MINE."""
    global mine, _mic_data
    mine = MINE()
    _mic_data = data

def init_candidate_mine(data=None, rows=None, cols=None):
    """Initializer for candidate-pair MIC workers."""
    global mine, _mic_data, _mic_candidate_rows, _mic_candidate_cols
    mine = MINE()
    _mic_data = data
    _mic_candidate_rows = rows
    _mic_candidate_cols = cols

def compute_mi_pair(args):
    """Compute MIC for a single pair of data points."""
    global mine
    if len(args) == 3:
        (i, j, data) = args
    else:
        (i, j) = args
        data = _mic_data
    if mine is None:
        mine = MINE()
    mine.compute_score(data[i, :], data[j, :])
    return (i, j, mine.mic())

def _pair_indices(num_cells):
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            yield (i, j)

def _row_offset(num_cells, row_idx):
    return row_idx * (num_cells - 1) - row_idx * (row_idx - 1) // 2

def _pair_blocks(num_cells, num_processes):
    target_pairs = max(1, num_cells * (num_cells - 1) // 2 // (num_processes * MIC_BLOCKS_PER_PROCESS))
    blocks = []
    start = 0
    count = 0
    for row_idx in range(num_cells - 1):
        count += num_cells - row_idx - 1
        if count >= target_pairs:
            blocks.append((start, row_idx + 1))
            start = row_idx + 1
            count = 0
    if start < num_cells - 1:
        blocks.append((start, num_cells - 1))
    return blocks

def compute_mi_block(args):
    """Compute MIC values for a contiguous block of upper-triangular rows."""
    global mine
    (row_start, row_end) = args
    data = _mic_data
    if mine is None:
        mine = MINE()
    num_cells = data.shape[0]
    num_values = sum((num_cells - i - 1 for i in range(row_start, row_end)))
    values = np.empty(num_values, dtype=np.float64)
    out_idx = 0
    for i in range(row_start, row_end):
        for j in range(i + 1, num_cells):
            mine.compute_score(data[i, :], data[j, :])
            values[out_idx] = mine.mic()
            out_idx += 1
    return (_row_offset(num_cells, row_start), values)

def compute_candidate_mi_block(args):
    """Compute MIC values for a contiguous block of candidate pairs."""
    global mine
    (start_idx, end_idx) = args
    data = _mic_data
    if mine is None:
        mine = MINE()
    values = np.empty(end_idx - start_idx, dtype=np.float64)
    for (out_idx, pair_idx) in enumerate(range(start_idx, end_idx)):
        i = int(_mic_candidate_rows[pair_idx])
        j = int(_mic_candidate_cols[pair_idx])
        mine.compute_score(data[i, :], data[j, :])
        values[out_idx] = mine.mic()
    return (start_idx, values)

def _contiguous_blocks(num_items, num_processes, blocks_per_process=MIC_BLOCKS_PER_PROCESS):
    target_items = max(1, num_items // (num_processes * blocks_per_process))
    blocks = []
    start = 0
    while start < num_items:
        end = min(num_items, start + target_items)
        blocks.append((start, end))
        start = end
    return blocks

def _upper_triangular_csr(values, num_cells):
    """Build the same upper-triangular CSR layout without materializing row ids."""
    indptr = np.empty(num_cells + 1, dtype=np.int64)
    indptr[0] = 0
    row_counts = np.arange(num_cells - 1, -1, -1, dtype=np.int64)
    np.cumsum(row_counts, out=indptr[1:])
    index_dtype = np.int64 if num_cells > np.iinfo(np.int32).max else np.int32
    indices = np.empty(values.size, dtype=index_dtype)
    offset = 0
    for i in range(num_cells - 1):
        count = num_cells - i - 1
        indices[offset:offset + count] = np.arange(i + 1, num_cells, dtype=index_dtype)
        offset += count
    return sp.csr_matrix((values, indices, indptr), shape=(num_cells, num_cells))

def _normalize_candidate_pairs(candidate_pairs, num_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return unique upper-triangular candidate pairs sorted by row and column.

    Accepted inputs are either a tuple of (rows, cols) arrays or an iterable of
    (row, col) pairs. Self-pairs are dropped.
    """
    if isinstance(candidate_pairs, tuple) and len(candidate_pairs) == 2:
        rows = np.asarray(candidate_pairs[0], dtype=np.int64)
        cols = np.asarray(candidate_pairs[1], dtype=np.int64)
    else:
        pairs = np.asarray(list(candidate_pairs), dtype=np.int64)
        if pairs.size == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError('candidate_pairs must contain (row, col) pairs.')
        rows = pairs[:, 0]
        cols = pairs[:, 1]
    if rows.shape != cols.shape:
        raise ValueError('candidate row and column arrays must have the same shape.')
    if rows.size == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    low = np.minimum(rows, cols)
    high = np.maximum(rows, cols)
    valid = (low != high) & (low >= 0) & (high < num_cells)
    if not np.all(valid):
        low = low[valid]
        high = high[valid]
    keys = low * num_cells + high
    keys = np.unique(keys)
    return (keys // num_cells, keys % num_cells)

def _all_upper_pairs(num_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(num_cells, k=1)

def _row_quantile_bins(data: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize each cell's feature ranks into equal-frequency row bins."""
    if n_bins < 2:
        raise ValueError('n_bins must be at least 2.')
    (n_cells, n_features) = data.shape
    order = np.argsort(data, axis=1, kind='mergesort')
    bins = np.empty((n_cells, n_features), dtype=np.int16)
    rank_bins = np.arange(n_features, dtype=np.int64) * n_bins // n_features
    rows = np.arange(n_cells)[:, None]
    bins[rows, order] = rank_bins[None, :]
    return bins

def _coarse_mi_score(data: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute a fast fixed-grid MI surrogate between cell profiles.

    This is not minepy's MIC. It is only used to choose candidate pairs whose
    final weights are still computed with stock minepy MIC.
    """
    data = np.asarray(data)
    (n_cells, n_features) = data.shape
    bins = _row_quantile_bins(data, n_bins)
    one_hot = np.zeros((n_bins, n_cells, n_features), dtype=np.float32)
    rows = np.arange(n_cells)[:, None]
    cols = np.arange(n_features)[None, :]
    one_hot[bins, rows, cols] = 1.0
    probabilities = np.empty((n_bins, n_cells), dtype=np.float32)
    for bin_idx in range(n_bins):
        probabilities[bin_idx] = one_hot[bin_idx].sum(axis=1) / n_features
    scores = np.zeros((n_cells, n_cells), dtype=np.float32)
    for left_bin in range(n_bins):
        px = probabilities[left_bin][:, None]
        for right_bin in range(n_bins):
            joint = one_hot[left_bin] @ one_hot[right_bin].T / n_features
            denom = px * probabilities[right_bin][None, :]
            nonzero = joint > 0
            term = np.zeros_like(joint)
            term[nonzero] = joint[nonzero] * np.log(joint[nonzero] / denom[nonzero])
            scores += term
    scores /= np.log(n_bins)
    np.fill_diagonal(scores, -np.inf)
    return scores

def _top_upper_pairs_from_score(score: np.ndarray, n_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
    n_cells = score.shape[0]
    (rows, cols) = np.triu_indices(n_cells, k=1)
    n_candidates = min(int(n_candidates), rows.size)
    if n_candidates <= 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    values = score[rows, cols]
    selected = np.argpartition(values, -n_candidates)[-n_candidates:]
    return (rows[selected], cols[selected])

def _top_rowwise_pairs_from_score(score: np.ndarray, candidates_per_row: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select the best coarse-MI neighbors for every cell, then de-duplicate."""
    n_cells = score.shape[0]
    candidates_per_row = min(max(0, int(candidates_per_row)), max(0, n_cells - 1))
    if candidates_per_row == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    work = np.asarray(score)
    cols = np.argpartition(work, -candidates_per_row, axis=1)[:, -candidates_per_row:]
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), candidates_per_row)
    cols = cols.reshape(-1).astype(np.int64, copy=False)
    return _normalize_candidate_pairs((rows, cols), n_cells)

def _top_up_pairs_from_score(score: np.ndarray, rows: np.ndarray, cols: np.ndarray, target_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fill unused candidate budget with the next best pairs from one score."""
    n_cells = score.shape[0]
    (rows, cols) = _normalize_candidate_pairs((rows, cols), n_cells)
    if rows.size >= target_candidates:
        return (rows, cols)
    (upper_rows, upper_cols) = np.triu_indices(n_cells, k=1)
    values = score[upper_rows, upper_cols]
    total_pairs = upper_rows.size
    existing_keys = set((rows * n_cells + cols).tolist())
    add_rows = []
    add_cols = []
    needed = target_candidates - rows.size
    keep = min(total_pairs, max(target_candidates, needed * 2))
    while len(add_rows) < needed:
        selected = np.argpartition(values, -keep)[-keep:]
        selected = selected[np.argsort(values[selected])[::-1]]
        for idx in selected:
            row = int(upper_rows[idx])
            col = int(upper_cols[idx])
            key = row * n_cells + col
            if key in existing_keys:
                continue
            existing_keys.add(key)
            add_rows.append(row)
            add_cols.append(col)
            if len(add_rows) == needed:
                break
        if len(add_rows) == needed or keep == total_pairs:
            break
        keep = min(total_pairs, max(keep * 2, keep + needed))
    if not add_rows:
        return (rows, cols)
    return _normalize_candidate_pairs((np.concatenate([rows, np.asarray(add_rows, dtype=np.int64)]), np.concatenate([cols, np.asarray(add_cols, dtype=np.int64)])), n_cells)

def _coarse_mi_score_block(one_hot: np.ndarray, probabilities: np.ndarray, start_idx: int, end_idx: int, upper_only: bool=True) -> np.ndarray:
    (n_bins, _, n_features) = one_hot.shape
    scores = np.zeros((end_idx - start_idx, one_hot.shape[1]), dtype=np.float32)
    for left_bin in range(n_bins):
        px = probabilities[left_bin, start_idx:end_idx, None]
        left = one_hot[left_bin, start_idx:end_idx]
        for right_bin in range(n_bins):
            joint = left @ one_hot[right_bin].T / n_features
            denom = px * probabilities[right_bin][None, :]
            nonzero = joint > 0
            term = np.zeros_like(joint)
            term[nonzero] = joint[nonzero] * np.log(joint[nonzero] / denom[nonzero])
            scores += term
    scores /= np.log(n_bins)
    if upper_only:
        for (local_row, row_idx) in enumerate(range(start_idx, end_idx)):
            scores[local_row, :row_idx + 1] = -np.inf
    else:
        for (local_row, row_idx) in enumerate(range(start_idx, end_idx)):
            scores[local_row, row_idx] = -np.inf
    return scores

def _top_upper_pairs_from_coarse_mi_blockwise(data: np.ndarray, n_bins: int, n_candidates: int, block_size: int, block_oversample: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select coarse-MI top pairs without materializing a full n by n score matrix.

    The final top-k is exact within the retained block candidates. Blocks keep a
    proportional oversampled candidate set so memory scales with n rather than
    n^2.
    """
    data = np.asarray(data)
    (n_cells, n_features) = data.shape
    bins = _row_quantile_bins(data, n_bins)
    one_hot = np.zeros((n_bins, n_cells, n_features), dtype=np.float32)
    rows = np.arange(n_cells)[:, None]
    cols = np.arange(n_features)[None, :]
    one_hot[bins, rows, cols] = 1.0
    probabilities = np.empty((n_bins, n_cells), dtype=np.float32)
    for bin_idx in range(n_bins):
        probabilities[bin_idx] = one_hot[bin_idx].sum(axis=1) / n_features
    block_rows = []
    block_cols = []
    block_values = []
    for start_idx in range(0, n_cells - 1, block_size):
        end_idx = min(n_cells - 1, start_idx + block_size)
        scores = _coarse_mi_score_block(one_hot, probabilities, start_idx, end_idx)
        finite = np.isfinite(scores)
        valid_count = int(finite.sum())
        if valid_count == 0:
            continue
        keep = int(np.ceil(n_candidates * (end_idx - start_idx) / n_cells * block_oversample))
        keep = min(max(1, keep), valid_count)
        flat_scores = scores.ravel()
        selected = np.argpartition(flat_scores, -keep)[-keep:]
        selected = selected[np.isfinite(flat_scores[selected])]
        local_rows = selected // n_cells
        selected_cols = selected % n_cells
        block_rows.append(local_rows + start_idx)
        block_cols.append(selected_cols)
        block_values.append(flat_scores[selected])
    if not block_rows:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    rows = np.concatenate(block_rows).astype(np.int64, copy=False)
    cols = np.concatenate(block_cols).astype(np.int64, copy=False)
    values = np.concatenate(block_values)
    (rows, cols) = _normalize_candidate_pairs((rows, cols), n_cells)
    if rows.size <= n_candidates:
        return (rows, cols)
    values = _score_selected_pairs(data, rows, cols, n_bins)
    selected = np.argpartition(values, -n_candidates)[-n_candidates:]
    return (rows[selected], cols[selected])

def _top_rowwise_pairs_from_coarse_mi_blockwise(data: np.ndarray, n_bins: int, candidates_per_row: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-bounded row-wise coarse-MI candidate selector."""
    data = np.asarray(data)
    (n_cells, n_features) = data.shape
    candidates_per_row = min(max(0, int(candidates_per_row)), max(0, n_cells - 1))
    if candidates_per_row == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    bins = _row_quantile_bins(data, n_bins)
    one_hot = np.zeros((n_bins, n_cells, n_features), dtype=np.float32)
    rows = np.arange(n_cells)[:, None]
    cols = np.arange(n_features)[None, :]
    one_hot[bins, rows, cols] = 1.0
    probabilities = np.empty((n_bins, n_cells), dtype=np.float32)
    for bin_idx in range(n_bins):
        probabilities[bin_idx] = one_hot[bin_idx].sum(axis=1) / n_features
    out_rows = []
    out_cols = []
    for start_idx in range(0, n_cells, block_size):
        end_idx = min(n_cells, start_idx + block_size)
        scores = _coarse_mi_score_block(one_hot, probabilities, start_idx, end_idx, upper_only=False)
        selected_cols = np.argpartition(scores, -candidates_per_row, axis=1)[:, -candidates_per_row:]
        out_rows.append(np.repeat(np.arange(start_idx, end_idx, dtype=np.int64), candidates_per_row))
        out_cols.append(selected_cols.reshape(-1).astype(np.int64, copy=False))
    return _normalize_candidate_pairs((np.concatenate(out_rows), np.concatenate(out_cols)), n_cells)

def _score_selected_pairs(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n_bins: int) -> np.ndarray:
    """Score selected pairs by fixed-grid MI without constructing all pair scores."""
    bins = _row_quantile_bins(data, n_bins)
    n_features = data.shape[1]
    scores = np.zeros(rows.size, dtype=np.float32)
    for left_bin in range(n_bins):
        left = bins[rows] == left_bin
        px = left.sum(axis=1) / n_features
        for right_bin in range(n_bins):
            right = bins[cols] == right_bin
            joint = (left & right).sum(axis=1) / n_features
            py = right.sum(axis=1) / n_features
            nonzero = joint > 0
            scores[nonzero] += joint[nonzero] * np.log(joint[nonzero] / (px[nonzero] * py[nonzero]))
    return scores / np.log(n_bins)

def build_mic_candidates(data, n_candidates: Optional[int]=None, candidates_per_cell: Optional[float]=200, n_bins: Sequence[int]=(3, 4), split_strategy: str='equal', max_dense_cells: int=20000, score_block_size: int=1024, block_oversample: float=2.0, fill_budget: bool=True, fill_bin: Optional[int]=None, selection_strategy: str='global') -> Tuple[np.ndarray, np.ndarray]:
    """
    Build candidate cell pairs for atlas-scale MIC calculation.

    The candidates are selected by a fast fixed-grid MI surrogate. The returned
    pairs should be passed to :func:`calculate_MIC_candidate`; their MIC values
    are not approximated.

    Parameters
    ----------
    data
        Cell by feature matrix, usually PCA scores.
    n_candidates
        Maximum number of undirected candidate pairs. If omitted, it is derived
        from ``candidates_per_cell``.
    candidates_per_cell
        Average undirected candidate pairs per cell. A value near 200 matched
        the high-recall setting on the pancreas benchmark.
    n_bins
        Coarse MI bin counts to ensemble. ``(3, 4)`` worked better than one
        fixed resolution at lower budgets.
    split_strategy
        Currently only ``"equal"`` is supported; the total candidate budget is
        split equally across the requested bin counts and then de-duplicated.
    max_dense_cells
        Use the exact dense score matrix up to this many cells. Larger inputs
        use a memory-bounded blockwise candidate selector.
    score_block_size
        Number of source rows per block in blockwise candidate selection.
    block_oversample
        Per-block oversampling factor before the final top-k merge.
    fill_budget
        If ``True``, fill candidate slots lost to ensemble duplicates using the
        next best pairs from ``fill_bin``. This preserves ensemble candidates
        while using the requested MIC budget more fully.
    fill_bin
        Coarse MI bin count used for top-up. Defaults to the last value in
        ``n_bins``.
    selection_strategy
        ``"global"`` keeps the highest coarse-MI pairs overall. ``"rowwise"``
        first keeps high-scoring neighbors per cell, which can preserve graph
        geometry better at the same MIC budget.
    """
    data = np.asarray(data)
    num_cells = data.shape[0]
    total_pairs = num_cells * (num_cells - 1) // 2
    if total_pairs == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if n_candidates is None:
        if candidates_per_cell is None:
            raise ValueError('Either n_candidates or candidates_per_cell must be provided.')
        n_candidates = int(np.ceil(num_cells * float(candidates_per_cell)))
    n_candidates = min(max(0, int(n_candidates)), total_pairs)
    if n_candidates == total_pairs:
        return _all_upper_pairs(num_cells)
    if split_strategy != 'equal':
        raise ValueError("Only split_strategy='equal' is currently supported.")
    if not n_bins:
        raise ValueError('n_bins must contain at least one bin count.')
    selection_strategy = selection_strategy.lower()
    if selection_strategy not in {'global', 'rowwise'}:
        raise ValueError("selection_strategy must be either 'global' or 'rowwise'.")
    per_score = int(np.ceil(n_candidates / len(n_bins)))
    per_row = int(np.ceil(float(n_candidates) / num_cells / len(n_bins)))
    candidate_rows = []
    candidate_cols = []
    dense_scores = {}
    for bin_count in n_bins:
        if num_cells <= max_dense_cells:
            score = _coarse_mi_score(data, int(bin_count))
            dense_scores[int(bin_count)] = score
            if selection_strategy == 'rowwise':
                (rows, cols) = _top_rowwise_pairs_from_score(score, per_row)
            else:
                (rows, cols) = _top_upper_pairs_from_score(score, per_score)
        elif selection_strategy == 'rowwise':
            (rows, cols) = _top_rowwise_pairs_from_coarse_mi_blockwise(data, int(bin_count), per_row, score_block_size)
        else:
            (rows, cols) = _top_upper_pairs_from_coarse_mi_blockwise(data, int(bin_count), per_score, score_block_size, block_oversample)
        candidate_rows.append(rows)
        candidate_cols.append(cols)
    rows = np.concatenate(candidate_rows) if candidate_rows else np.empty(0, dtype=np.int64)
    cols = np.concatenate(candidate_cols) if candidate_cols else np.empty(0, dtype=np.int64)
    (rows, cols) = _normalize_candidate_pairs((rows, cols), num_cells)
    if fill_budget and rows.size < n_candidates and (num_cells <= max_dense_cells):
        if fill_bin is None:
            fill_bin = int(n_bins[-1])
        if fill_bin not in dense_scores:
            dense_scores[fill_bin] = _coarse_mi_score(data, int(fill_bin))
        (rows, cols) = _top_up_pairs_from_score(dense_scores[fill_bin], rows, cols, n_candidates)
    return (rows, cols)

@calc_time
def calculate_MIC_candidate(data, candidate_pairs, num_processes=None):
    """
    Calculate stock minepy MIC only for selected candidate cell pairs.

    This keeps each computed MIC value identical to the exact all-pairs path for
    those pairs, but returns a sparse upper-triangular matrix containing only
    candidate edges.
    """
    data = np.asarray(data)
    num_cells = data.shape[0]
    (rows, cols) = _normalize_candidate_pairs(candidate_pairs, num_cells)
    num_pairs = rows.size
    if num_pairs == 0:
        return sp.csr_matrix((num_cells, num_cells), dtype=np.float64)
    if num_processes is None:
        num_processes = min(cpu_count(), AUTO_MAX_PROCESSES)
    num_processes = max(1, int(num_processes))
    values = np.empty(num_pairs, dtype=np.float64)
    if mic_pairs is not None:
        values = mic_pairs(data, rows, cols, n_threads=num_processes)
    elif num_processes == 1:
        local_mine = MINE()
        for (idx, (i, j)) in enumerate(zip(rows, cols)):
            local_mine.compute_score(data[int(i), :], data[int(j), :])
            values[idx] = local_mine.mic()
    else:
        with Pool(processes=num_processes, initializer=init_candidate_mine, initargs=(data, rows, cols)) as pool:
            for (offset, block_values) in pool.imap_unordered(compute_candidate_mi_block, _contiguous_blocks(num_pairs, num_processes)):
                values[offset:offset + block_values.size] = block_values
    return sp.csr_matrix((values, (rows, cols)), shape=(num_cells, num_cells))

@calc_time
def calculate_MIC_filtered(data, min_mic, num_processes=None, row_start=0, row_end=None):
    """
    Compute exact all-pairs MIC while retaining only values above a threshold.

    The optimized C backend compares each pair immediately and returns only
    retained triples, avoiding the quadratic condensed MIC allocation. A row
    range can be supplied for deterministic out-of-core or distributed shards;
    ``row_end`` is exclusive.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError('data must be a 2D cells x features array')
    num_cells = data.shape[0]
    min_mic = float(min_mic)
    if not np.isfinite(min_mic):
        raise ValueError('min_mic must be finite')
    row_start = int(row_start)
    if row_end is None:
        row_end = max(0, num_cells - 1)
    row_end = int(row_end)
    if not 0 <= row_start <= row_end <= max(0, num_cells - 1):
        raise ValueError('row range must satisfy 0 <= row_start <= row_end <= n_cells - 1')
    if num_cells < 2 or row_start == row_end:
        return sp.csr_matrix((num_cells, num_cells), dtype=np.float64)
    if num_processes is None:
        num_processes = min(cpu_count(), AUTO_MAX_THREADS)
    num_processes = max(1, int(num_processes))
    if mic_pstats_filtered is not None:
        (rows, cols, values) = mic_pstats_filtered(data, min_mic=min_mic, n_threads=num_processes, row_start=row_start, row_end=row_end)
    else:
        retained_rows = []
        retained_cols = []
        retained_values = []
        local_mine = MINE()
        for row in range(row_start, row_end):
            for col in range(row + 1, num_cells):
                local_mine.compute_score(data[row], data[col])
                value = local_mine.mic()
                if value >= min_mic:
                    retained_rows.append(row)
                    retained_cols.append(col)
                    retained_values.append(value)
        rows = np.asarray(retained_rows, dtype=np.int64)
        cols = np.asarray(retained_cols, dtype=np.int64)
        values = np.asarray(retained_values, dtype=np.float64)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(num_cells, num_cells))
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix

@calc_time
def calculate_MIC_atlas(data, num_processes=None, n_candidates: Optional[int]=None, candidates_per_cell: Optional[float]=200, n_bins: Sequence[int]=(3, 4), max_dense_cells: int=20000, score_block_size: int=1024, block_oversample: float=2.0, fill_budget: bool=True, fill_bin: Optional[int]=None, selection_strategy: str='global', return_candidates: bool=False):
    """
    Atlas-oriented approximate MIC matrix.

    Candidate pairs are selected by coarse MI, then each retained edge is scored
    by the original stock minepy MIC. The output is intentionally sparse and is
    not expected to be identical to all-pairs MIC.
    """
    (rows, cols) = build_mic_candidates(data, n_candidates=n_candidates, candidates_per_cell=candidates_per_cell, n_bins=n_bins, max_dense_cells=max_dense_cells, score_block_size=score_block_size, block_oversample=block_oversample, fill_budget=fill_budget, fill_bin=fill_bin, selection_strategy=selection_strategy)
    mic_matrix = calculate_MIC_candidate(data, (rows, cols), num_processes=num_processes)
    if return_candidates:
        return (mic_matrix, (rows, cols))
    return mic_matrix

@calc_time
def calculate_MIC(data, num_processes=None):
    """
    Calculate the MIC (Maximal Information Coefficient) matrix for the input data.

    Parameters:
    - data: 2D numpy array (cells x genes or PCA-reduced data)
    - num_processes: Number of parallel processes to use (default: CPU count)

    Returns:
    - mi_matrix: 2D symmetric MIC matrix (cells x cells)
    """
    data = np.asarray(data)
    num_cells = data.shape[0]
    num_pairs = num_cells * (num_cells - 1) // 2
    if num_pairs == 0:
        return sp.csr_matrix((num_cells, num_cells), dtype=np.float64)
    if num_processes is None:
        if mic_pstats is not None:
            num_processes = min(cpu_count(), AUTO_MAX_THREADS)
        else:
            num_processes = min(cpu_count(), AUTO_MAX_PROCESSES)
    num_processes = max(1, int(num_processes))
    values = np.empty(num_pairs, dtype=np.float64)
    if mic_pstats is not None:
        values = mic_pstats(data, n_threads=num_processes)
    elif num_processes == 1:
        local_mine = MINE()
        for (out_idx, (i, j)) in enumerate(_pair_indices(num_cells)):
            local_mine.compute_score(data[i, :], data[j, :])
            values[out_idx] = local_mine.mic()
    else:
        with Pool(processes=num_processes, initializer=init_mine, initargs=(data,)) as pool:
            for (offset, block_values) in pool.imap_unordered(compute_mi_block, _pair_blocks(num_cells, num_processes)):
                values[offset:offset + block_values.size] = block_values
    mic_matrix = _upper_triangular_csr(values, num_cells)
    return mic_matrix

def _prepare_data_for_pca(adata: AnnData, layer: str, use_highly_variable: bool, hv_kwargs: Dict, use_scale: bool) -> AnnData:
    """
    Prepares the AnnData object for PCA by optionally subsetting to highly variable genes and scaling data.

    Parameters:
    - adata: AnnData object
    - use_highly_variable: Whether to subset to highly variable genes
    - hv_kwargs: Keyword arguments for `sc.pp.highly_variable_genes`
    - use_scale: Whether to scale the data

    Returns:
    - adata: Prepared AnnData object
    """
    if hv_kwargs is None:
        hv_kwargs = {}
    if use_highly_variable:
        sc.pp.highly_variable_genes(adata, **hv_kwargs)
        logger.info(f'Found {adata.var.highly_variable.sum()} highly variable genes.')
    if use_scale:
        sc.pp.scale(adata, layer=layer)
        logger.info('Data scaled before PCA.')
    if use_highly_variable:
        if version.parse(sc.__version__) >= version.parse('1.10.0'):
            logger.info("Using `mask_var='highly_variable'` for PCA.")
            sc.pp.pca(adata, layer=layer, mask_var='highly_variable')
        else:
            logger.info('Using `use_highly_variable=True` for PCA (deprecated in scanpy 1.10.0).')
            sc.pp.pca(adata, layer=layer, use_highly_variable=True)
    else:
        sc.pp.pca(adata, layer=layer)
        logger.info('PCA performed without subsetting to HVGs.')
    logger.info('PCA completed.')
    return adata
logger = logging.getLogger(__name__)

@calc_time
def generate_mi_mat(adata, use_highly_variable: bool=False, hv_kwargs: Optional[Dict]=None, layer: Optional[str]=None, use_scale: bool=False, n_components: int=30, num_processes: Optional[int]=None, force_PCA: bool=False, mic_key: str='MIC', mic_mode: str='exact', n_candidates: Optional[int]=None, candidates_per_cell: Optional[float]=200, candidate_n_bins: Sequence[int]=(3, 4), candidate_fill_budget: bool=True, candidate_fill_bin: Optional[int]=None, candidate_selection_strategy: str='global', mic_min_threshold: Optional[float]=None):
    """
    Generate an MIC matrix and store it in the AnnData object.

    Parameters:
    - adata: AnnData object
    - use_highly_variable: Whether to use highly variable genes for PCA
    - hv_kwargs: Dictionary of keyword arguments for `sc.pp.highly_variable_genes`. Independent from 'layer' parameter.
    - layer: Which element of layers to use for scailing and PCA. Independent from 'hv_kwargs'. It means you can use different layer to perform PCA or find HVGs. If `None`, uses `adata.X`.
    - use_scale: Whether to scale the data before PCA
    - subset: Whether to subset to highly variable genes
    - n_components: Number of PCA components to use
    - num_processes: Number of processes for MIC calculation. If `None`, use all available CPUs.
    - force_PCA: Whether to force PCA even if already present
    - mic_key: Key to store the MIC matrix in `adata.obsp`
    - mic_mode: "exact", "exact_filtered", or approximate "atlas".
    - n_candidates: Number of candidate pairs in atlas mode.
    - candidates_per_cell: Candidate-pair budget per cell in atlas mode.
    - candidate_n_bins: Coarse MI bin counts used to select atlas candidates.
    - candidate_fill_budget: Fill unused candidate slots after ensemble de-duplication.
    - candidate_fill_bin: Coarse MI bin count used for candidate top-up.
    - candidate_selection_strategy: "global" or "rowwise" candidate selector.

    Returns:
    - adata: AnnData object with MIC matrix stored in `adata.obsp[mic_key]`
    """
    _adata = adata.copy()
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Specified layer '{layer}' does not exist in adata.layers.")
    if 'X_pca' in _adata.obsm and (not force_PCA):
        if layer is not None:
            logger.warning("⚠️ Layer '%s' specified but existing PCA matrix will be reused.", layer)
        logger.info('✅ X_pca already in adata.obsm. Using existing PCA matrix.')
        pca_mat = _adata.obsm['X_pca'][:, :n_components]
    else:
        if 'X_pca' in _adata.obsm and force_PCA:
            logger.info('✅ Force PCA is enabled. Recomputing PCA and overwriting existing X_pca.')
        else:
            logger.info('✅ Performing PCA with assigned parameters.')
        _adata = _prepare_data_for_pca(_adata, layer, use_highly_variable, hv_kwargs, use_scale)
        pca_mat = _adata.obsm['X_pca'][:, :n_components]
    if mic_mode == 'exact':
        logger.info('✅ Calculating exact all-pairs MIC matrix...')
        mi_matrix = calculate_MIC(pca_mat, num_processes=num_processes)
    elif mic_mode == 'exact_filtered':
        if mic_min_threshold is None:
            raise ValueError("mic_min_threshold is required for mic_mode='exact_filtered'.")
        logger.info('✅ Calculating exact all-pairs MIC with threshold %.6e...', mic_min_threshold)
        mi_matrix = calculate_MIC_filtered(pca_mat, min_mic=mic_min_threshold, num_processes=num_processes)
        adata.uns[f'{mic_key}_filtered_info'] = {'mode': 'exact_filtered', 'min_mic': float(mic_min_threshold), 'retained_pairs': int(mi_matrix.nnz)}
    elif mic_mode == 'atlas':
        logger.info('✅ Calculating atlas candidate MIC matrix...')
        (mi_matrix, candidate_pairs) = calculate_MIC_atlas(pca_mat, num_processes=num_processes, n_candidates=n_candidates, candidates_per_cell=candidates_per_cell, n_bins=candidate_n_bins, fill_budget=candidate_fill_budget, fill_bin=candidate_fill_bin, selection_strategy=candidate_selection_strategy, return_candidates=True)
        total_pairs = pca_mat.shape[0] * (pca_mat.shape[0] - 1) // 2
        adata.uns[f'{mic_key}_candidate_info'] = {'mode': 'atlas', 'candidate_pairs': int(candidate_pairs[0].size), 'total_pairs': int(total_pairs), 'candidate_fraction': float(candidate_pairs[0].size / total_pairs) if total_pairs else 0.0, 'n_candidates': None if n_candidates is None else int(n_candidates), 'candidates_per_cell': None if candidates_per_cell is None else float(candidates_per_cell), 'candidate_n_bins': [int(x) for x in candidate_n_bins], 'candidate_fill_budget': bool(candidate_fill_budget), 'candidate_fill_bin': None if candidate_fill_bin is None else int(candidate_fill_bin), 'candidate_selection_strategy': candidate_selection_strategy}
    else:
        raise ValueError("mic_mode must be 'exact', 'exact_filtered', or 'atlas'.")
    adata.obsp[mic_key] = mi_matrix
    logger.info('✅ MIC matrix stored in adata.obsp[%s].', mic_key)
    return adata
