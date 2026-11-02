from anndata import AnnData
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import numpy as np
from typing import List, Union

def _mannwhitneyu_less_rows(group_values, rest_values):
    """Vectorize scalar ``method='auto'`` decisions without changing p-values."""
    group_values = np.asarray(group_values)
    rest_values = np.asarray(rest_values)
    pvalues = np.empty(group_values.shape[0], dtype=np.float64)
    if min(group_values.shape[1], rest_values.shape[1]) > 8:
        pvalues[:] = mannwhitneyu(group_values, rest_values, axis=1, alternative='less', method='asymptotic').pvalue
        return pvalues
    combined = np.concatenate([group_values, rest_values], axis=1)
    combined.sort(axis=1)
    has_ties = np.any(combined[:, 1:] == combined[:, :-1], axis=1)
    for (selected, method) in ((~has_ties, 'exact'), (has_ties, 'asymptotic')):
        if not np.any(selected):
            continue
        pvalues[selected] = mannwhitneyu(group_values[selected], rest_values[selected], axis=1, alternative='less', method=method).pvalue
    return pvalues

class FatePredictor:

    def __init__(self, adata: AnnData, reld_key: str, group_by: str, source: str, target: List[str], copy: bool=False):
        """
        Initializes the FatePredictor class.

        Parameters:
        - adata: Input AnnData object.
        - reld_key: Key for the relative dimension matrix in obsp. This matrix represents the relative dimensions between cells.
        - group_by: Column in obs specifying cluster annotations.
        - source: Name of the source cluster.
        - target: List of target cluster names.
        - copy: Whether to copy the input AnnData object.
        """
        self.adata = adata.copy() if copy else adata
        self.reld_key = reld_key
        self.group_by = group_by
        self.source = source
        self.target = target

    @classmethod
    def from_selected_rd(cls, adata: AnnData, reld_matrix, group_by: str, source: str, target: List[str], source_indices=None, target_indices=None, copy: bool=False):
        """Create a predictor from an exact source-by-target RD subset."""
        predictor = cls(adata, reld_key='__selected_rd__', group_by=group_by, source=source, target=target, copy=copy)
        if group_by not in predictor.adata.obs:
            raise ValueError(f"'{group_by}' not found in `adata.obs`. Please check the column name.")
        labels = predictor.adata.obs[group_by]
        if source_indices is None:
            source_indices = np.flatnonzero(labels.to_numpy() == source)
        else:
            source_indices = np.asarray(source_indices, dtype=np.int64)
        if target_indices is None:
            target_indices = np.flatnonzero(labels.isin(target).to_numpy())
        else:
            target_indices = np.asarray(target_indices, dtype=np.int64)
        if reld_matrix.shape != (source_indices.size, target_indices.size):
            raise ValueError('reld_matrix shape must match source_indices x target_indices')
        reld_adata = AnnData(X=reld_matrix.copy() if copy else reld_matrix)
        reld_adata.obs_names = predictor.adata.obs_names[source_indices]
        reld_adata.var_names = predictor.adata.obs_names[target_indices]
        reld_adata.obs[group_by] = source
        reld_adata.var[group_by] = predictor.adata.obs.iloc[target_indices][group_by].to_numpy()
        predictor.reld_adata = reld_adata
        predictor.source_numeric = source_indices
        predictor.target_numeric = target_indices
        return predictor

    def _validate_inputs(self):
        """Validates the input parameters."""
        if self.reld_key not in self.adata.obsp:
            raise ValueError(f"'{self.reld_key}' not found in `adata.obsp`. Please check your AnnData object.")
        if self.group_by not in self.adata.obs:
            raise ValueError(f"'{self.group_by}' not found in `adata.obs`. Please check the column name.")
        if self.source not in self.adata.obs[self.group_by].unique():
            raise ValueError(f"Source cluster '{self.source}' not found in the '{self.group_by}' column.")
        if not all((target_cluster in self.adata.obs[self.group_by].unique() for target_cluster in self.target)):
            raise ValueError('One or more target clusters are not found in the dataset.')
        if len(self.target) < 2:
            raise ValueError('Please specify at least two target clusters for comparison.')

    def _create_anndata(self):
        """
        Creates a new AnnData object by subsetting the input adata based on source and target indices.
        """
        self._validate_inputs()
        source_indices = self.adata.obs[self.adata.obs[self.group_by] == self.source].index
        target_indices = self.adata.obs[self.adata.obs[self.group_by].isin(self.target)].index
        source_numeric = self.adata.obs_names.get_indexer(source_indices)
        target_numeric = self.adata.obs_names.get_indexer(target_indices)
        reld = self.adata.obsp[self.reld_key]
        reld_subset = reld[source_numeric.reshape(-1, 1), target_numeric]
        new_adata = AnnData(X=reld_subset)
        new_adata.obs_names = source_indices
        new_adata.var_names = target_indices
        new_adata.obs[self.group_by] = self.source
        new_adata.var[self.group_by] = self.adata.obs.loc[target_indices, self.group_by]
        self.reld_adata = new_adata
        print(f'New AnnData created for statistical testing: {len(source_indices)} source × {len(target_indices)} target.')

    def _resolve_tie(self, na_value):
        """
        Resolves ties in predicted fate by performing pairwise Mann-Whitney U tests.
        If there are exactly two clusters in `predicted_fate`, it determines the fate based on the same criteria.
        Adds a 'resolve_tie' column indicating whether a tie was resolved.
        """
        if 'resolve_tie' not in self.reld_adata.obs:
            self.reld_adata.obs['resolve_tie'] = False
        for row_idx in tqdm(self.reld_adata.obs.index, desc='Resolving ties...'):
            row_fates = self.reld_adata.obs.loc[row_idx, 'predicted_fate']
            if len(row_fates) == 2:
                (cluster_a, cluster_b) = row_fates
                mask_a = self.reld_adata.var[self.group_by] == cluster_a
                mask_b = self.reld_adata.var[self.group_by] == cluster_b
                values_a = self.reld_adata[row_idx, mask_a].X.toarray().flatten()
                values_b = self.reld_adata[row_idx, mask_b].X.toarray().flatten()
                values_a[values_a == 0] = na_value
                values_b[values_b == 0] = na_value
                (_, pvalue) = mannwhitneyu(values_a, values_b, alternative='two-sided')
                if pvalue < 0.05 and np.median(values_a) < np.median(values_b):
                    self.reld_adata.obs.at[row_idx, 'predicted_fate'] = [cluster_a]
                    self.reld_adata.obs.at[row_idx, 'resolve_tie'] = True
                elif pvalue < 0.05 and np.median(values_b) < np.median(values_a):
                    self.reld_adata.obs.at[row_idx, 'predicted_fate'] = [cluster_b]
                    self.reld_adata.obs.at[row_idx, 'resolve_tie'] = True

    def predict_fate(self, resolve_tie: bool=True, na_value: Union[float, str]='max', use_fdr: bool=True, res_to_string: bool=True):
        """
        Performs statistical tests on the clusters in var for each row in reld_adata.
        Adds p-values and adjusted p-values for each cluster as new columns in reld_adata.obs.
        Also adds a 'predicted_fate' column based on the specified criteria.

        Parameters:
        - resolve_tie: Whether to resolve ties in the predicted fate using Mann-Whitney U test between tied clusters.
        - na_value: Value to replace 0s with during the test. If "max", replaces 0s with a value slightly larger than the maximum non-zero value.
                     It's crucial to replace zeros with non-zero values because zero values may not accurately represent the absence of a biological relationship.
                     Instead, they often represent data that couldn't be captured due to technical limitations, potentially leading to false positives or negatives in statistical tests.
        - use_fdr: Whether to use FDR-adjusted p-values for determining the predicted fate.
        """
        if not hasattr(self, 'reld_adata'):
            print('reld_adata not found. Automatically creating reld_adata for further testing.')
            self._create_anndata()
        if self.group_by not in self.reld_adata.var:
            raise ValueError(f"'{self.group_by}' column must be in reld_adata.var for testing.")
        if na_value == 'max':
            non_zero_max = self.reld_adata.X.data.max() if sp.issparse(self.reld_adata.X) and np.any(self.reld_adata.X.data != 0) else np.max(self.reld_adata.X[self.reld_adata.X != 0], initial=0)
            if non_zero_max > 0:
                na_value = non_zero_max + non_zero_max * 0.01
            else:
                na_value = 1
            print(f'na_value is set to {na_value}')
        clusters = self.reld_adata.var[self.group_by].unique()
        cluster_masks = {cluster: (self.reld_adata.var[self.group_by] == cluster).values for cluster in clusters}
        p_values = np.zeros((self.reld_adata.n_obs, len(clusters)))
        for (cluster_idx, cluster) in enumerate(clusters):
            group_mask = cluster_masks[cluster]
            rest_mask = ~group_mask
            group_values = self.reld_adata.X[:, group_mask].toarray()
            rest_values = self.reld_adata.X[:, ~group_mask].toarray()
            for row_idx in range(self.reld_adata.n_obs):
                group_row = group_values[row_idx].flatten()
                rest_row = rest_values[row_idx].flatten()
                group_row[group_row == 0] = na_value
                rest_row[rest_row == 0] = na_value
                (_, pvalue) = mannwhitneyu(group_row, rest_row, alternative='less')
                p_values[row_idx, cluster_idx] = pvalue
        for (cluster_idx, cluster) in enumerate(clusters):
            self.reld_adata.obs[f'{cluster}_p'] = p_values[:, cluster_idx]
        print('Initial testing is complete, and p-values are added to reld_adata.obs.')
        for (cluster_idx, cluster) in enumerate(clusters):
            raw_pvalues = p_values[:, cluster_idx]
            (_, fdr_corrected_pvalues, _, _) = multipletests(raw_pvalues, method='fdr_bh')
            self.reld_adata.obs[f'{cluster}_adj_p'] = fdr_corrected_pvalues
        print('FDR-adjusted p-values are added to reld_adata.obs.')
        predicted_fate = []
        FDR_OR_NOT = 'adjusted p-values' if use_fdr else 'p-values'
        print(f'{FDR_OR_NOT} are used for classification.')
        for row_idx in tqdm(self.reld_adata.obs.index, desc='Classifying...'):
            row_fates = []
            for (cluster_idx, cluster) in enumerate(clusters):
                group_mask = cluster_masks[cluster]
                group_values = self.reld_adata[row_idx, group_mask].X.toarray().flatten()
                rest_values = self.reld_adata[row_idx, ~group_mask].X.toarray().flatten()
                group_values[group_values == 0] = na_value
                rest_values[rest_values == 0] = na_value
                p_value_key = f'{cluster}_adj_p' if use_fdr else f'{cluster}_p'
                if self.reld_adata.obs.loc[row_idx, p_value_key] < 0.05 and np.median(group_values) < np.median(rest_values):
                    row_fates.append(cluster)
            predicted_fate.append(row_fates)
        self.reld_adata.obs['predicted_fate'] = predicted_fate
        if resolve_tie and len(clusters) > 2:
            self._resolve_tie(na_value)

        def list_to_string(lst):
            if not lst:
                return 'Undetermined'
            else:
                return ' & '.join(lst)
        if res_to_string:
            self.reld_adata.obs['predicted_fate'] = self.reld_adata.obs['predicted_fate'].apply(list_to_string)
        return None

    def predict_fate_streaming(self, resolve_tie: bool=True, na_value: Union[float, str]='max', use_fdr: bool=True, res_to_string: bool=True, row_batch_size: int=256, disable_tqdm: bool=False):
        """
        Predict fate with bounded dense memory while preserving test semantics.

        Only ``row_batch_size x n_targets`` values are densified at once. The
        Mann-Whitney calls, group order, FDR correction, median criterion, and
        tie resolution are the same as in :meth:`predict_fate`.
        """
        if not hasattr(self, 'reld_adata'):
            self._create_anndata()
        if self.group_by not in self.reld_adata.var:
            raise ValueError(f"'{self.group_by}' column must be in reld_adata.var for testing.")
        row_batch_size = int(row_batch_size)
        if row_batch_size < 1:
            raise ValueError('row_batch_size must be at least 1')
        matrix = self.reld_adata.X
        if na_value == 'max':
            if sp.issparse(matrix):
                non_zero_max = matrix.data.max() if np.any(matrix.data != 0) else 0
            else:
                values = np.asarray(matrix)
                non_zero_max = np.max(values[values != 0], initial=0)
            na_value = non_zero_max + non_zero_max * 0.01 if non_zero_max > 0 else 1
            print(f'na_value is set to {na_value}')
        else:
            na_value = float(na_value)
        clusters = self.reld_adata.var[self.group_by].unique()
        cluster_masks = {cluster: (self.reld_adata.var[self.group_by] == cluster).to_numpy() for cluster in clusters}
        n_rows = self.reld_adata.n_obs
        p_values = np.zeros((n_rows, len(clusters)), dtype=np.float64)
        median_less = np.zeros((n_rows, len(clusters)), dtype=bool)
        ranges = range(0, n_rows, row_batch_size)
        for start in tqdm(ranges, desc='Testing fate in row blocks...', disable=disable_tqdm):
            end = min(n_rows, start + row_batch_size)
            if sp.issparse(matrix):
                block = matrix[start:end].toarray()
            else:
                block = np.asarray(matrix[start:end]).copy()
            block[block == 0] = na_value
            for (cluster_idx, cluster) in enumerate(clusters):
                group_mask = cluster_masks[cluster]
                group_values = block[:, group_mask]
                rest_values = block[:, ~group_mask]
                p_values[start:end, cluster_idx] = _mannwhitneyu_less_rows(group_values, rest_values)
                median_less[start:end, cluster_idx] = np.median(group_values, axis=1) < np.median(rest_values, axis=1)
        for (cluster_idx, cluster) in enumerate(clusters):
            self.reld_adata.obs[f'{cluster}_p'] = p_values[:, cluster_idx]
            (_, corrected, _, _) = multipletests(p_values[:, cluster_idx], method='fdr_bh')
            self.reld_adata.obs[f'{cluster}_adj_p'] = corrected
        selected_pvalues = np.column_stack([self.reld_adata.obs[f'{cluster}_adj_p' if use_fdr else f'{cluster}_p'].to_numpy() for cluster in clusters])
        significant = (selected_pvalues < 0.05) & median_less
        predicted_fate = [[cluster for (cluster_idx, cluster) in enumerate(clusters) if significant[row_idx, cluster_idx]] for row_idx in range(n_rows)]
        self.reld_adata.obs['predicted_fate'] = predicted_fate
        if resolve_tie and len(clusters) > 2:
            self.reld_adata.obs['resolve_tie'] = False
            for start in tqdm(range(0, n_rows, row_batch_size), desc='Resolving ties in row blocks...', disable=disable_tqdm):
                end = min(n_rows, start + row_batch_size)
                if sp.issparse(matrix):
                    block = matrix[start:end].toarray()
                else:
                    block = np.asarray(matrix[start:end]).copy()
                block[block == 0] = na_value
                for local_row in range(end - start):
                    row_idx = start + local_row
                    row_fates = self.reld_adata.obs.iloc[row_idx]['predicted_fate']
                    if len(row_fates) != 2:
                        continue
                    (cluster_a, cluster_b) = row_fates
                    values_a = block[local_row, cluster_masks[cluster_a]].flatten()
                    values_b = block[local_row, cluster_masks[cluster_b]].flatten()
                    (_, pvalue) = mannwhitneyu(values_a, values_b, alternative='two-sided')
                    if pvalue < 0.05 and np.median(values_a) < np.median(values_b):
                        self.reld_adata.obs.iat[row_idx, self.reld_adata.obs.columns.get_loc('predicted_fate')] = [cluster_a]
                        self.reld_adata.obs.iat[row_idx, self.reld_adata.obs.columns.get_loc('resolve_tie')] = True
                    elif pvalue < 0.05 and np.median(values_b) < np.median(values_a):
                        self.reld_adata.obs.iat[row_idx, self.reld_adata.obs.columns.get_loc('predicted_fate')] = [cluster_b]
                        self.reld_adata.obs.iat[row_idx, self.reld_adata.obs.columns.get_loc('resolve_tie')] = True
        if res_to_string:
            self.reld_adata.obs['predicted_fate'] = self.reld_adata.obs['predicted_fate'].apply(lambda values: 'Undetermined' if not values else ' & '.join(values))
        return None
