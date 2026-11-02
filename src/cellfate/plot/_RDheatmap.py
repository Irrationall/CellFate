from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from ._plot_utils import create_handles

def RDheatmap(adata: AnnData, obsp_key: str='RD', sort_by: str='predicted_fate', color_mappings: Optional[Dict[str, List[str]]]=None, palette: Optional[Dict[str, str]]=None, cmap: str='viridis', cbar_label: str='Relative Dimensions', figsize: Tuple[int, int]=(10, 10), dendrogram_ratio: float=0.11, row_cluster: bool=False, col_cluster: bool=False, save_path: Optional[str]=None, show_plot: bool=True) -> Optional[plt.Figure]:
    """
    Generates a clustermap from the given AnnData object, replacing NaN values with the maximum value if present,
    otherwise replacing zeros with the maximum value.

    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the data.
    obsp_key : str, optional
        The key in adata.obsp to use for the matrix, by default 'RD'.
    sort_by : str, optional
        The column in adata.obs to sort the data by, by default 'predicted_fate'.
    color_mappings : dict, optional
        A dictionary specifying which columns to map to colors. Example:
        {
            'row': ['label', 'predicted_fate'],
            'col': ['annotation']
        }
    palette : dict, optional
        A dictionary specifying color palettes for each categorical variable. Example:
        {
            'label': 'Accent',
            'predicted_fate': 'tab10',
            'annotation': 'husl'
        }
    cmap : str, optional
        The colormap for the clustermap, by default 'viridis'.
    cbar_label : str, optional
        The label for the color bar, by default 'relative values'.
    figsize : tuple, optional
        Size of the figure, by default (14, 12).
    dendrogram_ratio : float, optional
        The ratio for dendrogram space, by default 0.11.
    row_cluster: bool, optional
        Whether to cluster rows. Defaults to False.
    col_cluster: bool, optional
        Whether to cluster columns. Defaults to False.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    show_plot : bool, optional
        Whether to display the plot, by default True.

    Returns:
    --------
    Optional[plt.Figure]
        The matplotlib figure object if show_plot is False, otherwise None.
    """
    if obsp_key not in adata.obsp:
        raise KeyError(f"'{obsp_key}' not found in adata.obsp")
    if sort_by not in adata.obs.columns:
        raise KeyError(f"'{sort_by}' not found in adata.obs")
    matrix = adata.obsp[obsp_key].toarray()
    if np.isnan(matrix).any():
        max_value = np.nanmax(matrix)
        filtered_matrix = np.where(np.isnan(matrix), max_value, matrix)
    else:
        max_value = np.max(matrix)
        filtered_matrix = np.where(matrix == 0, max_value, matrix)
    sorted_idx = adata.obs[sort_by].sort_values().index
    sorted_idx_positions = np.array([adata.obs.index.get_loc(i) for i in sorted_idx])
    sorted_data = filtered_matrix[sorted_idx_positions, :][:, sorted_idx_positions]
    if color_mappings is None:
        color_mappings = {'row': ['label', 'predicted_fate'], 'col': ['annotation']}
    row_colors = []
    col_colors = []
    legends = []
    for key in color_mappings.get('row', []):
        unique_values = adata.obs[key].astype(str).unique()
        palette_name = palette.get(key, 'tab10') if palette else 'tab10'
        colors = sns.color_palette(palette_name, len(unique_values))
        color_dict = {val: colors[i] for (i, val) in enumerate(unique_values)}
        row_color = adata.obs.loc[sorted_idx, key].astype(str).map(color_dict).to_numpy()
        row_colors.append(row_color)
        handles = create_handles(unique_values, color_dict, marker='o')
        legends.append((f'Row: {key}', handles, len(unique_values)))
    for key in color_mappings.get('col', []):
        unique_values = adata.obs[key].astype(str).unique()
        palette_name = palette.get(key, 'tab10') if palette else 'tab10'
        colors = sns.color_palette(palette_name, len(unique_values))
        color_dict = {val: colors[i] for (i, val) in enumerate(unique_values)}
        col_color = adata.obs.loc[sorted_idx, key].astype(str).map(color_dict).to_numpy()
        col_colors.append(col_color)
        handles = create_handles(unique_values, color_dict, marker='s')
        legends.append((f'Col: {key}', handles, len(unique_values)))
    g = sns.clustermap(sorted_data, row_cluster=row_cluster, col_cluster=col_cluster, annot=False, cmap=cmap, cbar_kws={'label': cbar_label}, row_colors=row_colors if row_colors else None, col_colors=col_colors if col_colors else None, figsize=figsize, dendrogram_ratio=dendrogram_ratio)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    fig = g.fig
    start_y = 0.9
    min_spacing = 0.02
    additional_spacing_per_label = 0.018
    previous_legend_length = 0
    for (title, handles, num_labels) in legends:
        dynamic_spacing = min_spacing + previous_legend_length * additional_spacing_per_label
        y_pos = start_y - dynamic_spacing
        bbox = (1.0, y_pos)
        legend = fig.legend(handles=handles, labels=[handle.get_label() for handle in handles], title=title, loc='upper left', bbox_to_anchor=bbox, borderaxespad=0.0, frameon=True, edgecolor='black', facecolor='white', fontsize='small', markerscale=1.2)
        fig.add_artist(legend)
        start_y = y_pos - 0.02
        previous_legend_length = num_labels
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        return fig
