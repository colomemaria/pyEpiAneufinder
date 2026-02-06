import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

import anndata as ad

def split_subclones(res, split_val, criterion="maxclust",
                    dist_metric="euclidean", linkage_method="ward"):
    """
    Function to split the CNV results into subclones based on hierarchical clustering

    Parameters
    ----------
    res: Results from pyEpiAneufinder main function (as pandas data frame)
    split_val: Either number of clones to split the data (for criterion = maxclust) or cluster distance (for criterion = distanc)
    criterion: Either maxclust (splitting tree into a certain number of clones defined in split_val) or
               distance (splitting tree based on a certain distance defined in split_val)
    dist_metric: Distance metric between CNV profile (e.g. euclidean, cityblock)
    linkage_method: Linkage method for hierarchical clustering (e.g. Ward, complete, average)

    Returns
    ------
    Pandas DataFrame with two columns of barcode and subclone group
    """

    # Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq","start","end"])

    # Calculate pairwise distances between cells
    dist_matrix = pdist(data_matrix.T, metric=dist_metric)

    # Normalize the distance by the number of bins (=> mean absolute error)
    fract_deviation = dist_matrix / res.shape[0]

    # Hierarchical clustering
    hc_cluster = linkage(fract_deviation, method=linkage_method)

    # Split into groups
    cl_members = fcluster(Z=hc_cluster, t=split_val, criterion=criterion)
    
    clones = pd.DataFrame({"barcode": data_matrix.columns.values,
                           "subclone": cl_members})

    return clones



def plot_single_cell_profile(outdir, cell_name, plot_path):
    """
    Function to visualize the count distribution (GC corrected) per somy for one selected cell

    Parameters
    ----------
    outdir: Directory containing results from pyEpiAneufinder main function
    cell_name: Barcode of selected cell
    plot_path: Path to saved the distribution plots

    Returns
    ------
    Collection of distribution plots
    """

    # Read both result table and count matrix
    res = pd.read_csv(outdir + "/result_table.csv", index_col=0)
    counts = ad.read(outdir + "/count_matrix.h5ad")

    # Check that the cell name is found in the data frame
    if not (cell_name in res.columns):
        raise ValueError(f"Cell barcode {cell_name} not in the result data frame!")

    # Collect data
    plot_data = pd.DataFrame({"chr": res["seq"],
                              "gc_counts": counts.X[counts.obs.cellID == cell_name].toarray().flatten(),
                              "somy": res[cell_name]})
    
    # Remove extreme quantiles (1% and 99.9%)
    plot_data = plot_data[(plot_data.gc_counts > plot_data.gc_counts.quantile(0.01)) &
                          (plot_data.gc_counts < plot_data.gc_counts.quantile(0.999))]
    
    # Get a numeric column to position each bin within the genome
    plot_data["pos"] = range(len(plot_data))

    # Convert into text
    plot_data["somy_text"] = plot_data.somy.map({0: "loss", 
                                                 1: "base", 
                                                 2: "gain"})

    # Add a smoothed mean line as additional estimation
    plot_data["counts_smooth"] = plot_data["gc_counts"].rolling(window=200, center=True).mean()

    # Create a density plot over the somies
    custom_palette = {"loss": "#9A32CD", "base": "#00EE76", "gain": "#CD0000"}

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
    sns.kdeplot(data=plot_data, x="gc_counts", hue="somy_text", 
            common_norm=False, fill=True, bw_adjust=1,
            hue_order=["loss", "base", "gain"],
            palette=custom_palette, ax=ax_kde)

    ax_kde.legend_.set_title("")
    ax_kde.set_xlabel("Counts per bin - GC corrected")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"{cell_name}: Library size - {len(plot_data)}")

    # --------------------------------------------------------------------------------------------------
    # Summary of somy occurance
    # --------------------------------------------------------------------------------------------------

    # Estimate occurrences in pandas and put into right format
    somy_counts = plot_data['somy_text'].value_counts().reindex(['gain', 'base', 'loss'], fill_value=0)
    somy_counts_list = list(somy_counts.items())

    ax_table = fig.add_subplot(top_gs[0, 1])
    ax_table.axis("off")

    table = ax_table.table(cellText=somy_counts_list,
                        colLabels=["Somy", "# bins"],
                        cellLoc='center',
                        loc='center')

    table.scale(1, 1.5)
    table.set_fontsize(12)

    # --------------------------------------------------------------------------------------------------
    # Summary of somy occurance
    # --------------------------------------------------------------------------------------------------

    bottom_gs = outer_gs[1].subgridspec(2, len(relative_size), width_ratios=relative_size, hspace=0.1)

    for i, chr_name in enumerate(chromosomes):

        subset = grouped.get_group(chr_name)

        # Create a first plot with the rolling mean function
        ax = fig.add_subplot(bottom_gs[0, i])
        ax.scatter(subset["pos"],subset["gc_counts"], color="grey",alpha=0.5)
        ax.plot(subset["pos"],subset["counts_smooth"],color="red")

        ax.set_title(chr_name, rotation=30)

        ax.set_ylim(total_min, total_max)

        # Remove the axes ticks and labels
        ax.set_xticks([])
        ax.set_xlabel('')

        if(i!= 0):
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel('GC counts')

        # Create a second plot with somy annotations
        ax = fig.add_subplot(bottom_gs[1, i])


        sns.scatterplot(data=subset, x="pos", y="gc_counts", hue="somy_text", palette=custom_palette, 
                        alpha=0.7, ax=ax, legend=False)

        ax.set_ylim(total_min, total_max)

        # Remove the axes ticks and labels
        ax.set_xticks([])
        ax.set_xlabel('')

        if(i!= 0):
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel('GC counts')

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')