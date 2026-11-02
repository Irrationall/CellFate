from anndata import AnnData
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ._plot_utils import darken_color
from typing import List

def volcanoplot(adata: AnnData, group: str, pct_cutoff: float=0.1, logfc_threshold: float=0.5, p_threshold: float=0.05, use_fdr: bool=True, color_up: str='#d66e5f', color_down: str='#5fc7d6', alpha_sig: float=0.3, genes_to_annotate_up: List=None, genes_to_annotate_down: List=None, annot_fontsize: float=10, figsize: tuple=(8, 6), return_fig: bool=False, show: bool=True, save: str=None):
    if 'rank_genes_groups' not in adata.uns:
        raise ValueError("The 'rank_genes_groups' result is missing from `adata.uns`. Please run `sc.tl.rank_genes_groups(adata, groupby='your_grouping_variable')` first.")
    df = sc.get.rank_genes_groups_df(adata, group=group)
    df = df[df['pct_nz_group'] >= pct_cutoff]
    if use_fdr:
        df['neg_log10_pvals'] = -np.log10(df['pvals_adj'] + 1e-300)
    else:
        df['neg_log10_pvals'] = -np.log10(df['pvals'] + 1e-300)
    p_threshold_log = -np.log10(p_threshold)
    darker_color_up = darken_color(color_up, amount=0.2)
    darker_color_down = darken_color(color_down, amount=0.2)
    (fig, ax) = plt.subplots(figsize=figsize)
    scatter_plot = ax.scatter(df['logfoldchanges'], df['neg_log10_pvals'], c='#99a3a4', edgecolor='grey', alpha=0.1)
    ax.scatter(df.loc[(df['logfoldchanges'] > logfc_threshold) & (df['neg_log10_pvals'] > p_threshold_log), 'logfoldchanges'], df.loc[(df['logfoldchanges'] > logfc_threshold) & (df['neg_log10_pvals'] > p_threshold_log), 'neg_log10_pvals'], c=color_up, alpha=alpha_sig, edgecolor='black')
    ax.scatter(df.loc[(df['logfoldchanges'] < -logfc_threshold) & (df['neg_log10_pvals'] > p_threshold_log), 'logfoldchanges'], df.loc[(df['logfoldchanges'] < -logfc_threshold) & (df['neg_log10_pvals'] > p_threshold_log), 'neg_log10_pvals'], c=color_down, alpha=alpha_sig, edgecolor='black')
    texts = []
    if genes_to_annotate_up:
        for gene in genes_to_annotate_up:
            if gene in df['names'].values:
                gene_row = df[df['names'] == gene].iloc[0]
                texts.append(ax.text(gene_row['logfoldchanges'], gene_row['neg_log10_pvals'], gene, fontsize=annot_fontsize, color=darker_color_up, ha='right', va='center'))
    if genes_to_annotate_down:
        for gene in genes_to_annotate_down:
            if gene in df['names'].values:
                gene_row = df[df['names'] == gene].iloc[0]
                texts.append(ax.text(gene_row['logfoldchanges'], gene_row['neg_log10_pvals'], gene, fontsize=annot_fontsize, color=darker_color_down, ha='left', va='center'))
    ax.set_xlabel('Log2 Fold Change', fontsize=20, weight='bold', labelpad=20)
    ylabel = '-Log10 Adjusted P-value' if use_fdr else '-Log10 P-value'
    ax.set_ylabel(ylabel, fontsize=20, weight='bold', labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.axhline(y=p_threshold_log, color='black', linestyle='--', linewidth=1.2)
    ax.axvline(x=logfc_threshold, color='black', linestyle='--', linewidth=1.2)
    ax.axvline(x=-logfc_threshold, color='black', linestyle='--', linewidth=1.2)
    if return_fig:
        return (fig, ax)
    else:
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)
            return (fig, ax)
