import pandas as pd
import numpy as np
import anndata as ad
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def karyo_gainloss(res, outdir, title=None, annot_dt=None,
                   state_type='categorical', n_states=5, 
                   linkage_method='ward', dist_metric='euclidean'):
    """
    Function to plot karyogram with support for categorical, integer or continuous CN states

    Parameters
    ----------
    res: Pandas DataFrame with position information in first three columns (seq, start, end), 
         followed by the CN status of each cell
    outdir: Path to saved .png image
    title: Plot title
    annot_dt: Optional Pandas DataFrame with annotation information for each cell. 
        Index must match cell barcodes in res. Must contain column 'annot' with categorical annotations.
    state_type: Type of states to plot. Options: 'categorical', 'integer', 'continuous'
    n_states: Number of categorical states (3 or 5). Only used when state_type='categorical'
    linkage_method: Linkage method for hierarchical clustering. Default: 'ward'
    dist_metric: Distance metric for clustering. Default: 'euclidean'

    Returns
    -------
    A filtered version of the input Pandas DataFrame

    """

    # ----------------------------
    # Input validation
    # ----------------------------
    valid_state_types = ['categorical', 'integer', 'continuous']
    if state_type not in valid_state_types:
        raise ValueError(f"state_type must be one of {valid_state_types}")

    for col in ['seq', 'start', 'end']:
        if col not in res.columns:
            raise KeyError(f"Missing required column: {col}")

    if res.shape[0] == 0:
        raise ValueError("Input DataFrame is empty")

    data_matrix = res.drop(columns=["seq", "start", "end"])

    if data_matrix.shape[0] == 0 or data_matrix.shape[1] == 0:
        raise ValueError("Data matrix is empty after dropping position columns")

    if not all(pd.api.types.is_numeric_dtype(data_matrix[col]) for col in data_matrix.columns):
        raise ValueError("All CNV columns must be numeric")

    if state_type == 'categorical':
        if n_states not in [3, 5]:
            raise ValueError("n_states must be 3 or 5 for categorical state_type")
        valid_values = {3: {0, 1, 2}, 5: {0, 0.5, 1, 1.5, 2}}[n_states]
        if not set(np.unique(data_matrix.values)).issubset(valid_values):
            raise ValueError("Unexpected categorical CNV values detected")

    # ----------------------------
    # Clustering
    # ----------------------------
    dist_matrix = pdist(data_matrix.T, metric=dist_metric)
    Z = linkage(dist_matrix, method=linkage_method)

    # Chromosome layout
    chr_num_bins = res["seq"].value_counts(sort=False)
    perc_chr = [c / sum(chr_num_bins) for c in chr_num_bins]
    res['seq'] = pd.Categorical(res['seq'], categories=chr_num_bins.index, ordered=True)

    # ----------------------------
    # Figure & GridSpec
    # ----------------------------
    fig = plt.figure(figsize=(22, 8))

    if annot_dt is None:
        gs = gridspec.GridSpec(
            1, len(perc_chr) + 1,
            width_ratios=[2] + [20 * p for p in perc_chr]
        )
    else:
        # sanity check
        if set(data_matrix.columns).difference(annot_dt.index):
            raise ValueError("Annotation DataFrame does not match cell barcodes")

        gs = gridspec.GridSpec(
            1, len(perc_chr) + 2,
            width_ratios=[2] + [20 * p for p in perc_chr] + [0.5]
        )

    # ----------------------------
    # Dendrogram
    # ----------------------------
    ax = fig.add_subplot(gs[0, 0])
    dendro = dendrogram(Z, orientation="left",
                        link_color_func=lambda k: 'darkgrey', ax=ax)
    ax.axis('off')

    leaf_order = dendro['leaves']
    leaf_order = [l + 3 for l in leaf_order[::-1]]
    res = res.iloc[:, [0, 1, 2] + leaf_order]

    if annot_dt is not None:
        barcodes_order = data_matrix.columns[dendro['leaves'][::-1]]
        annot_dt = annot_dt.loc[barcodes_order].reset_index()

    # ----------------------------
    # Colormaps
    # ----------------------------
    if state_type == 'categorical':
        if n_states == 3:
            cmap = ["#9A32CD", "#00EE76", "#CD0000"]
            legend_elements = [
                Patch(facecolor="#9A32CD", label="Loss"),
                Patch(facecolor="#00EE76", label="Base"),
                Patch(facecolor="#CD0000", label="Gain")
            ]
        else:
            cmap = ["#9A32CD", "#D7A0E8", "#00EE76", "#F08080", "#CD0000"]
            legend_elements = [
                Patch(facecolor="#9A32CD", label="Loss"),
                Patch(facecolor="#D7A0E8", label="Putative Loss"),
                Patch(facecolor="#00EE76", label="Base"),
                Patch(facecolor="#F08080", label="Putative Gain"),
                Patch(facecolor="#CD0000", label="Gain")
            ]
        vmin, vmax = 0, 2

    else:
        base_cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)
        shifted_cmap = shiftedColorMap(base_cmap, midpoint=1/3)
        cbar_label = 'Integer state' if state_type == 'integer' else 'Continuous score'

    # ----------------------------
    # Per-chromosome heatmaps
    # ----------------------------
    chromosome_groups = list(res.groupby('seq', observed=True))

    for i, (seq, group) in enumerate(chromosome_groups):
        ax = fig.add_subplot(gs[0, i + 1])
        data_filtered = group.drop(columns=["seq", "start", "end"])

        if state_type == 'categorical':
            sns.heatmap(data_filtered.T, ax=ax, cmap=cmap,
                        vmin=vmin, vmax=vmax, cbar=False)
        else:
            sns.heatmap(data_filtered.T, ax=ax, cmap=shifted_cmap,
                        vmin=0, vmax=6, 
                        cbar=False)

        ax.set_title(seq, rotation=30)
        ax.set_xticks([])
        ax.set_yticks([])

    # ----------------------------
    # Annotation bar
    # ----------------------------
    if annot_dt is not None:

        #Convert annot column automatically to category if not done
        if annot_dt["annot"].dtype.name != "category":
            print("Automatically converting column \"annot\" to categorical.")
            annot_dt["annot"] = annot_dt["annot"].astype("category")

        annot_numeric = annot_dt["annot"].cat.codes.values.reshape(-1, 1)
        labels_annot = annot_dt["annot"].cat.categories

        palette_annot = sns.color_palette("Set2", len(labels_annot))
        ax_annot = fig.add_subplot(gs[0, -1])

        sns.heatmap(annot_numeric, ax=ax_annot,
                    cmap=palette_annot, cbar=False,
                    xticklabels=False, yticklabels=False)

        ax_annot.set_title("Label", rotation=30)

        annot_legend = [
            Patch(facecolor=palette_annot[i], label=str(label))
            for i, label in enumerate(labels_annot)
        ]

        fig.legend(handles=annot_legend, loc='lower right',
                   ncol=len(annot_legend), frameon=False,
                   bbox_to_anchor=(1.0, -0.06), fontsize=12)

    # ----------------------------
    # Legends & layout
    # ----------------------------
    fig.text(0.5, 0.0, 'Position in chromosome', ha='center', va='center', fontsize=12)
    
    if state_type == 'categorical':
        fig.legend(handles=legend_elements, loc='upper right', 
                   ncol=len(legend_elements), frameon=False,
                   bbox_to_anchor=(1.0, 1.05), fontsize=12)
    if state_type in ['integer', 'continuous']:
        # Create inset axis in figure coordinates (legend-like)
        cax = inset_axes(
            ax,
            width="10%",
            height="1%",
            loc="upper right",
            bbox_to_anchor=(0, 0.06, 1, 1),
            bbox_transform=fig.transFigure,
            borderpad=0
        )

        sm = plt.cm.ScalarMappable(
            cmap=shifted_cmap,
            norm=plt.Normalize(vmin=0, vmax=6)
        )
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([0, 2, 6])
        cbar.set_ticklabels(['0', '2', '6'])

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outdir, dpi=300, bbox_inches='tight')



def plot_single_cell_profile(outdir, cell_name, plot_path, mode=None):
    """
    Function to visualize the count distribution (GC corrected) per somy for one selected cell

    Parameters
    ----------
    outdir: Directory containing results from pyEpiAneufinder's main function
    cell_name: Barcode of selected cell
    plot_path: Path to saved .png plot
    mode: Choose from "holmes" or "watson". If mode=None, use 5-state combined output.

    Returns
    ------
    Collection of distribution plots
    """

    # Read both result table and count matrix
    if mode:
        res = pd.read_csv(outdir + f"/outs/result_table_{mode}.tsv.gz", index_col=0, sep="\t")
    else:
        res = pd.read_csv(outdir + f"/outs/result_table.tsv.gz", index_col=0, sep="\t")
    counts = ad.read_h5ad(outdir + "/count_matrix.h5ad")

    # Check that the cell name is found in the data frame
    if not (cell_name in res.columns):
        raise ValueError(f"Cell barcode {cell_name} not in the result data frame!")

    # Collect data
    plot_data = pd.DataFrame({"chr": res["seq"],
                              "gc_counts": counts[counts.obs.cellID == cell_name].X.toarray().flatten(),
                              "somy": res[cell_name]})
    
    # Remove extreme quantiles (1% and 99.9%)
    # plot_data = plot_data[(plot_data.gc_counts > plot_data.gc_counts.quantile(0.01)) &
    #                       (plot_data.gc_counts < plot_data.gc_counts.quantile(0.999))]
    
    # Get a numeric column to position each bin within the genome
    plot_data["pos"] = range(len(plot_data))

    # Convert into text
    if mode:
        plot_data["somy_text"] = plot_data.somy.map({0: "loss",
                                                     1: "base",
                                                     2: "gain"})
    else:
        plot_data["somy_text"] = plot_data.somy.map({0: "loss",
                                                     0.5: "putative loss",
                                                     1: "base",
                                                     1.5: "putative gain",
                                                     2: "gain"})

    # Add a smoothed mean line as additional estimation
    plot_data["counts_smooth"] = plot_data["gc_counts"].rolling(window=200, center=True).mean()

    # Create a density plot over the somies
    if mode:
        custom_palette = {"loss": "#9A32CD", "base": "#00EE76", "gain": "#CD0000"}
    else:
        custom_palette = {"loss": "#9A32CD", 
                          "putative loss": "#D7A0E8", 
                          "base": "#00EE76", 
                          "putative gain": "#F08080", 
                          "gain": "#CD0000"}

    # Group data by chromosome and get counts (number of points)
    grouped = plot_data.groupby("chr")
    chromosomes = plot_data.chr.unique()
    counts = [len(grouped.get_group(chr)) for chr in chromosomes]

    # Get dimensions for the subplots
    relative_size = [c / sum(counts) * 30 for c in counts]
    total_max = plot_data["gc_counts"].max()
    total_min = plot_data["gc_counts"].min()

    # --------------------------------------------------------------------------------------------------
    # Create plot (overall three parts)
    # --------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(max(sum(relative_size), 12), 12))
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 3], hspace=0.3)

    top_gs = outer_gs[0].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # --------------------------------------------------------------------------------------------------
    # Density plot of the somy counts
    # --------------------------------------------------------------------------------------------------

    # Remark: different bandwidth and kernel than R function
    ax_kde = fig.add_subplot(top_gs[0, 0])
    if mode:
        sns.kdeplot(data=plot_data, x="gc_counts", hue="somy_text", 
                common_norm=False, fill=True, bw_adjust=3,
                hue_order=["loss", "base", "gain"],
                palette=custom_palette, ax=ax_kde)
    else:
        sns.kdeplot(data=plot_data, x="gc_counts", hue="somy_text", 
            common_norm=False, fill=True, bw_adjust=2.5,
            hue_order=["loss", "putative loss", "base", "putative gain", "gain"],
            palette=custom_palette, ax=ax_kde)

    ax_kde.legend_.set_title("")
    for text in ax_kde.legend_.get_texts():
        text.set_fontsize(16)

    ax_kde.tick_params(axis='both', labelsize=14)
    ax_kde.set_xlabel("GC corrected counts per bin", fontsize=16)
    ax_kde.set_ylabel("Density", fontsize=16)
    ax_kde.set_title(f"{cell_name}: Library size - {len(plot_data)}", fontsize=20)

    sns.despine(ax=ax_kde, top=True, right=True)

    # --------------------------------------------------------------------------------------------------
    # Summary of somy occurance
    # --------------------------------------------------------------------------------------------------

    # Estimate occurrences in pandas and put into right format
    if mode:
        somy_counts = plot_data['somy_text'].value_counts().reindex(['gain', 'base', 'loss'], fill_value=0)
    else:
        somy_counts = plot_data['somy_text'].value_counts().reindex(['gain', 'putative gain', 'base', 'putative loss', 'loss'], fill_value=0)

    somy_counts_list = list(somy_counts.items())

    ax_table = fig.add_subplot(top_gs[0, 1])
    ax_table.axis("off")

    table = ax_table.table(
        cellText=somy_counts_list,
        colLabels=["Somy", "# bins"],
        cellLoc='left',
        loc='upper right'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header row
            cell.set_text_props(weight='bold', fontsize=18)

    # --------------------------------------------------------------------------------------------------
    # Bottom panel: single continuous genome axis
    # --------------------------------------------------------------------------------------------------

    bottom_gs = outer_gs[1].subgridspec(
        3, 1,
        height_ratios=[0.2, 1, 1],
        hspace=0.05
    )

    # --- Prepare chromosome bin counts ---
    chr_bin_counts = plot_data["chr"].value_counts(sort=False)
    chr_bin_counts = chr_bin_counts.loc[chromosomes]  # preserve order

    genome_end = plot_data["pos"].max()

    # --------------------------------------------------------------------------------------------------
    # 1) Chromosome bar (top strip)
    # --------------------------------------------------------------------------------------------------

    ax_chr = fig.add_subplot(bottom_gs[0])

    start = 0
    for i, (chromosome, count) in enumerate(chr_bin_counts.items()):
        end = start + count

        color = "lightgrey" if i % 2 == 0 else "grey"
        ax_chr.add_patch(
            patches.Rectangle((start, 0), count, 1, color=color)
        )

        label = chromosome.replace("chr", "")
        ax_chr.text(
            (start + end) / 2,
            1.4,
            label,
            ha="center",
            va="center",
            fontsize=14
        )

        start = end

    ax_chr.set_xlim(0, genome_end)
    ax_chr.set_ylim(0, 2)
    ax_chr.axis("off")
    ax_chr.margins(x=0)


    # --------------------------------------------------------------------------------------------------
    # 2) Rolling mean + raw counts
    # --------------------------------------------------------------------------------------------------

    ax_raw = fig.add_subplot(bottom_gs[1])

    sns.scatterplot(
        data=plot_data,
        x="pos",
        y="gc_counts",
        color="grey",
        edgecolor=None,
        alpha=0.6,
        legend=False,
        ax=ax_raw
    )

    ax_raw.plot(
        plot_data["pos"],
        plot_data["counts_smooth"],
        color="red",
        linewidth=1
    )

    ax_raw.set_ylim(total_min, total_max)
    ax_raw.set_xlim(0, genome_end)
    ax_raw.tick_params(axis='y', labelsize=14)
    ax_raw.set_xlabel("")
    ax_raw.set_ylabel("GC counts", fontsize=16)
    ax_raw.set_xticks([])
    ax_raw.margins(x=0)

    # Add chromosome boundary lines
    start = 0
    for chromosome, count in chr_bin_counts.items():
        start += count
        ax_raw.axvline(start, color="black", linestyle="--", linewidth=0.75)


    # --------------------------------------------------------------------------------------------------
    # 3) Somy-colored scatter
    # --------------------------------------------------------------------------------------------------

    ax_somy = fig.add_subplot(bottom_gs[2], sharex=ax_raw)

    sns.scatterplot(
        data=plot_data,
        x="pos",
        y="gc_counts",
        hue="somy_text",
        palette=custom_palette,
        edgecolor=None,
        alpha=0.7,
        legend=False,
        ax=ax_somy
    )

    ax_somy.set_ylim(total_min, total_max)
    ax_somy.set_xlim(0, genome_end)
    ax_somy.tick_params(axis='y', labelsize=14)
    ax_somy.set_ylabel("GC counts", fontsize=16)
    ax_somy.set_xlabel("Genomic bins", fontsize=16, visible=True)
    ax_somy.xaxis.label.set_visible(True)
    ax_somy.margins(x=0)

    # Add chromosome boundary lines
    start = 0
    for chromosome, count in chr_bin_counts.items():
        start += count
        ax_somy.axvline(start, color="black", linestyle="--", linewidth=0.75)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. 
    
    Input
    -----
      cmap : Matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Should be between 0.0 and 1.0. Default: 0
      midpoint : The new center of the colormap. Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin)). Default: 0.5 (no shift)
      stop : Offset from highets point in the colormap's range.
          Should be between 0.0 and 1.0. Default: 1.0

    Returns
    -------
      newcmap : The new shifted colormap
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
      
    # Regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # Shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
        
    newcmap = mcolors.LinearSegmentedColormap(name, cdict)

    return newcmap