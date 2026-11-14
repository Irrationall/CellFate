from functools import lru_cache
from warnings import warn

from anndata import AnnData
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.path import Path
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


@lru_cache(maxsize=128)
def _wedge_marker(count, index):
    """One clockwise segment of a circular scatter marker, starting at 3 o'clock."""
    angles = np.linspace(index, index + 1, 32) * (-2 * np.pi / count)
    vertices = np.vstack(((0, 0), np.column_stack((np.cos(angles), np.sin(angles))), (0, 0)))
    return Path(vertices, [Path.MOVETO] + [Path.LINETO] * len(angles) + [Path.CLOSEPOLY])


def plot_volcano_multicolor(
    df, name_col="names", logfc_col="logfoldchanges", pval_col="pvals_adj",
    gene_sets=None, slice_order=None, threshold_logfc=0.5, pval_alpha=0.05,
    annotate_only_if_sig=False, marker_size=160, bg_point_size=10,
    figsize=(12, 8), xlim=None, ylim=None, color_all="#99a3a4",
    color_edge="grey", color_up="#d66e5f", color_down="#5fc7d6",
    alpha_bg=0.10, alpha_sig=0.22, halo_linewidth=3, label_fontsize=16,
    axis_label_fontsize=35, tick_fontsize=30, spine_width=3, show_legend=True,
    legend_loc="upper right", legend_title="Sets", explicit_ticks=True,
    xtick_step=1.0, ytick_step=5.0, adjust_text_kwargs=None, title=None,
    savepath=None, save_dpi=300, transparent=False, return_texts=False,
):
    """Plot a differential-expression table and highlight genes by set.

    `df` contains gene names, log2 fold changes and adjusted p-values in
    `name_col`, `logfc_col` and `pval_col`. Zero p-values are drawn at
    -log10 of the smallest positive float64. Both axes autoscale by default.
    Supply xlim and/or ylim to zoom. Points outside the requested view are
    counted once in a warning, not annotated or removed from the table.
    Rows without finite coordinates are excluded from these counts.

    Supply sets as e.g. `{"Markers": {"genes": ["Col3a1"], "color": "teal"}}`.
    A gene in multiple sets gets one clockwise slice per membership, in
    `slice_order` (default: mapping order). If given, it must list every set
    exactly once. The label takes the first membership color.
    With duplicate names the last row anchors the label.
    `annotate_only_if_sig` requires both abs(logFC) > threshold_logfc and
    p < pval_alpha. Pass a dict to replace the default adjustText options.

    Returns (fig, ax), or (fig, ax, texts) when return_texts=True. An optional
    savepath writes the figure without showing or closing it.
    """
    missing = [col for col in (name_col, logfc_col, pval_col) if col not in df]
    if missing:
        raise ValueError(f"Missing volcano plot columns: {missing}")
    if not 0 < pval_alpha <= 1 or not np.isfinite(threshold_logfc) or threshold_logfc < 0:
        raise ValueError("Require 0 < pval_alpha <= 1 and a finite, nonnegative threshold_logfc.")
    if explicit_ticks and not all(np.isfinite(step) and step > 0 for step in (xtick_step, ytick_step)):
        raise ValueError("Explicit tick steps must be finite and positive.")

    x = df[logfc_col].to_numpy(dtype=float, na_value=np.nan)
    p = df[pval_col].to_numpy(dtype=float, na_value=np.nan)
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"{pval_col} must contain probabilities between 0 and 1.")
    y = -np.log10(np.where(p == 0, np.nextafter(0., 1.), p))
    cutoff = -np.log10(pval_alpha)
    significant = (np.abs(x) > threshold_logfc) & (y > cutoff)

    sets = {} if gene_sets is None else gene_sets
    order = list(sets if slice_order is None else slice_order)
    omitted = [name for name in sets if name not in order]
    unknown = [name for name in order if name not in sets]
    if omitted or unknown or len(order) != len(set(order)):
        raise ValueError("slice_order must list every gene set exactly once; "
                         f"missing={omitted}, unknown={unknown}, "
                         f"duplicates={len(order) != len(set(order))}")
    rows = {gene: i for i, gene in enumerate(df[name_col])}
    memberships = {}
    for name, spec in sets.items():
        for gene in spec.get("genes", []):
            memberships.setdefault(gene, set()).add(name)
    absent = [gene for gene in memberships if gene not in rows]
    if absent:
        warn(f"Not found in {name_col}: {absent}", UserWarning, stacklevel=2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c=color_all, edgecolor=color_edge, alpha=alpha_bg, s=bg_point_size, zorder=0)
    guide = dict(color="black", linestyle="--", linewidth=1.2, zorder=0.5)
    ax.axhline(cutoff, **guide)
    for boundary in (threshold_logfc, -threshold_logfc):
        ax.axvline(boundary, **guide)
    for mask, color in ((significant & (x > 0), color_up), (significant & (x < 0), color_down)):
        ax.scatter(x[mask], y[mask], c=color, alpha=alpha_sig, edgecolor="none", s=12, zorder=0.6)

    # Resolve automatic limits after plotting all data, before placing labels.
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    xbounds, ybounds = ax.get_xlim(), ax.get_ylim()
    xmin, xmax = sorted(xbounds)
    ymin, ymax = sorted(ybounds)
    finite = np.isfinite(x) & np.isfinite(y)
    visible = finite & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    outside = np.count_nonzero(finite & ~visible)
    if outside and (xlim is not None or ylim is not None):
        warn(f"{outside} of {finite.sum()} finite points outside the requested view "
             "(not displayed or labeled).", UserWarning, stacklevel=2)

    texts = []
    halo = [patheffects.withStroke(linewidth=halo_linewidth, foreground="white")]
    for gene, member in memberships.items():
        if gene not in rows:
            continue
        row = rows[gene]
        if not visible[row] or (annotate_only_if_sig and not significant[row]):
            continue
        colors = [sets[name]["color"] for name in order if name in member]
        coords = (x[row], y[row])
        point = dict(s=marker_size, edgecolor="black")
        if len(colors) > 1:
            for i, color in enumerate(colors):
                ax.scatter(*coords, **point, marker=_wedge_marker(len(colors), i),
                           facecolor=color, linewidth=0.6, zorder=6 + i * 0.01)
        ax.scatter(*coords, **point, marker="o", linewidth=0.9,
                   facecolor=colors[0] if len(colors) == 1 else "none",
                   zorder=6 if len(colors) == 1 else 6.5)
        texts.append(ax.text(*coords, gene, color=colors[0], fontsize=label_fontsize,
                             fontweight="bold", ha="center", va="center", zorder=7, path_effects=halo))

    axis_style = dict(fontsize=axis_label_fontsize, weight="bold", labelpad=20)
    ax.set_xlabel("Log2 Fold Change", **axis_style)
    ax.set_ylabel("-Log10 (Adjusted P-value)", **axis_style)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if explicit_ticks:
        for low, high, step, setter in ((xmin, xmax, xtick_step, ax.set_xticks),
                                        (max(0, ymin), ymax, ytick_step, ax.set_yticks)):
            # Leave the automatic locator in place for extreme spans or a
            # close-up that contains no multiple of the requested tick step.
            first, last = np.ceil(low / step), np.floor(high / step)
            if 0 <= last - first < 1000:
                setter(np.arange(first, last + 1) * step)
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    if show_legend and order:
        handles = [Line2D([], [], marker="o", linestyle="None", markersize=10,
                          markerfacecolor=sets[name]["color"], markeredgecolor="black",
                          label=name) for name in order]
        ax.legend(handles=handles, title=legend_title, frameon=True, loc=legend_loc)
    if title:
        ax.set_title(title, fontsize=axis_label_fontsize, weight="bold", pad=10)
    fig.tight_layout()
    # Repulsion uses display coordinates: finish layout before moving the labels.
    if texts:
        from adjustText import adjust_text

        if adjust_text_kwargs is None:
            adjust_text_kwargs = dict(
                force_text=(0.2, 0.2), force_static=(0.3, 0.3), force_explode=(0.1, 0.1),
                expand=(1.5, 2.5), explode_radius=32, pull_threshold=15,
                ensure_inside_axes=True, only_move={"text": "xy", "explode": "xy", "pull": "xy"},
                avoid_self=True, arrowprops=dict(arrowstyle="->", color="black", lw=1,
                                                  shrinkA=0, shrinkB=0, mutation_scale=10),
            )
        adjust_text(texts, ax=ax, **adjust_text_kwargs)
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
    if savepath:
        fig.savefig(savepath, dpi=save_dpi, bbox_inches="tight", transparent=transparent)
    return (fig, ax, texts) if return_texts else (fig, ax)
