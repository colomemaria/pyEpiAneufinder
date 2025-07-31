import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

import anndata as ad

def split_subclones(res, num_clust):
    """
    Function to split the CNV results into subclones based on hierarchical clustering

    Parameters
    ----------
    res: Results from pyEpiAneufinder main function (as pandas data frame)
    num_clust: Number of clones to split the data

    Output
    ------
    Pandas data frame with two columns of barcode and subclone group
    """

    #Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq","start","end"])

    #Calculate pairwise distances between cells
    dist_matrix = pdist(data_matrix.T,metric="euclidean")

    #Hierarchical clustering
    hc_cluster = linkage(dist_matrix, method='ward')

    #Split into groups
    cl_members = fcluster(Z=hc_cluster, t=num_clust, criterion='maxclust')
    
    clones = pd.DataFrame({"barcode":data_matrix.columns.values,
                           "subclone":cl_members})

    return clones



def plot_karyo_annotated(res, plot_path, annot_dt=None, title_karyo=""):
    """
    Function to recreate the karyogram plot - optional with an annotation side bar

    Parameters
    ----------
    res: Results from pyEpiAneufinder main function (as pandas data frame)
    plot_path: path to save the karyogram
    annot_dt: option to add an annotation side bar, requires a pandas DataFrame 
              with barcodes as index and one column called "annot" as a categorical variable
    title_karyo: Option to give a plot title 

    Output
    ------
    Karyogram plot saved at plot_path
    """
    
    #Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq","start","end"])

    #Calculate pairwise distances between cells
    dist_matrix = pdist(data_matrix.T,metric="euclidean")

    #Hierarchical clustering
    Z = linkage(dist_matrix, method='ward')

    #Get the relative percentage of entries per chromosome
    chr_num_bins=res["seq"].value_counts(sort=False)
    perc_chr = [chr / sum(chr_num_bins) for chr in chr_num_bins]

    # Make sure the chromosomes are sorted in the correct order in the plots
    res['seq'] = pd.Categorical(res['seq'], categories=chr_num_bins.index, ordered=True)

    # Create a figure with a custom GridSpec layout
    fig = plt.figure(figsize=(22, 8))

    #Define size (dependent if an annotation column should be added)
    if annot_dt is None:
        gs = gridspec.GridSpec(1, len(perc_chr)+1, width_ratios=[2]+[20 * p for p in perc_chr])
    else:
        
        #Check that set of barcodes is matching
        if set(data_matrix.columns).difference(annot_dt.barcode):
            raise ValueError(f"The annotation DataFrame contains not the same cells as in the results matrix!")
        gs = gridspec.GridSpec(1, len(perc_chr)+2, width_ratios=[2]+[20 * p for p in perc_chr]+[0.5])

    #Add the dendogram as the first column
    ax = fig.add_subplot(gs[0, 0])
    dendro = dendrogram(Z,orientation="left", link_color_func=lambda k: 'darkgrey',ax=ax)
    #Remove the axes
    ax.axis('off')

    #Reorder the data matrix correctly
    leaf_order = dendro['leaves']  # Extract the order of the leaves as a list
    leaf_order = [l + 3 for l in leaf_order[::-1]] # Reverse the order and add + 3 (first three columns)
    res = res.iloc[:, [0,1,2]+leaf_order]

    if annot_dt is not None:
        barcodes_order = data_matrix.columns[dendro['leaves'][::-1]]
        annot_dt = annot_dt.loc[barcodes_order].reset_index()

    # Iterate over each unique 'seq' and plot in the respective grid
    for i, (seq, group) in enumerate(res.groupby('seq', observed=True)):
        ax = fig.add_subplot(gs[0, i+1])  # Add a subplot to the GridSpec
        data_filtered = group.drop(columns=["seq", "start", "end"])
        
        sns.heatmap(data_filtered.T, ax=ax, cmap=["#9A32CD", "#00EE76", "#CD0000"], vmin=0, vmax=2, cbar=False)
        ax.set_title(seq, rotation=30)
        
        #Remove the axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
     
    #Add a heatmap with annotations in the last subfigure if provided
    if annot_dt is not None:
        
        annot_numeric = annot_dt["annot"].cat.codes.values.reshape(-1, 1)
        labels_annot = annot_dt["annot"].cat.categories
        
        #Define the color palette
        palette_annot = sns.color_palette("Set2",len(labels_annot))
        ax_annot = fig.add_subplot(gs[0, -1])
        sns.heatmap(annot_numeric, cbar=False, cmap=palette_annot, xticklabels=False, yticklabels=False)
        ax_annot.set_title("Annot", rotation=30)
        
        #Create a legend for the annotation frame
        annot_dict = dict(zip(labels_annot, palette_annot))
        
        annot_legend = [Patch(facecolor=annot_dict[label], edgecolor='black', label=str(label)) 
                        for label in labels_annot]
        
        fig.legend(handles=annot_legend, loc='lower center', ncol= len(annot_legend),
                fontsize=12, frameon=False,bbox_to_anchor=(0.5, -0.08))

    # Add a common x-axis label
    fig.text(0.5, 0.0, 'Position in chromosome', ha='center', va='center', fontsize=12)

    # Add a legend below all plots for loss, base, gain
    legend_elements = [Patch(facecolor="#9A32CD", edgecolor='black', label='Loss'),
                       Patch(facecolor="#00EE76", edgecolor='black', label='Base'),
                       Patch(facecolor="#CD0000", edgecolor='black', label='Gain')]

    #Place legend dependent on whether there is a second legend from the annotation data frame
    if annot_dt is not None:
        fig.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=12, frameon=False,
                   bbox_to_anchor=(1.0, 1.02))
    else:
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, frameon=False,
                   bbox_to_anchor=(0.5, -0.08)) 

    # Adjust layout
    plt.subplots_adjust(bottom=0.2)

    # Show the plot
    plt.suptitle(title_karyo)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


def plot_single_cell_profile(outdir, cell_name, plot_path):
    """
    Function to visualize the count distribution (GC corrected) per somy for one selected cell

    Parameters
    ----------
    outdir: Results from pyEpiAneufinder main function (as pandas data frame)
    cell_name: barcode of the selected cell
    plot_path: path to save the distribution plots

    Output
    ------
    Collection of distribution plots saved at plot_path
    """

    #Read both result table and count matrix
    res = pd.read_csv(outdir + "/result_table.csv",index_col=0)
    counts = ad.read(outdir + "/count_matrix.h5ad")

    #Check that the cell name is found in the data frame
    if not (cell_name in res.columns):
        raise ValueError(f"Cell barcode {cell_name} not in the result data frame!")

    #Collect data
    plot_data = pd.DataFrame({"chr":res["seq"],
                              "gc_counts": counts.X[counts.obs.cellID == cell_name].toarray().flatten(),
                              "somy": res[cell_name]})
    
    #Remove extreme quantiles (1% and 99.9%)
    plot_data = plot_data[(plot_data.gc_counts > plot_data.gc_counts.quantile(0.01)) &
                          (plot_data.gc_counts < plot_data.gc_counts.quantile(0.999))]
    
    #Get a numeric column to position each bin within the genome
    plot_data["pos"]=range(len(plot_data))

    #Convert into text
    plot_data["somy_text"] = plot_data.somy.map({0:"loss",1:"base",2:"gain"})

    #Add a smoothed mean line as additional estimation
    plot_data["counts_smooth"] = plot_data["gc_counts"].rolling(window=200, center=True).mean()

    #Create a density plot over the somies
    custom_palette = {"loss": "#9A32CD", "base": "#00EE76", "gain": "#CD0000"}

    # Group data by chromosome and get counts (number of points)
    grouped = plot_data.groupby("chr")
    chromosomes = plot_data.chr.unique()
    counts = [len(grouped.get_group(chr)) for chr in chromosomes]

    #Get dimensions for the subplots
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

    #Remark: different bandwidth and kernel than R function
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

    #Estimate occurances in pandas and put into right format
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

        #Create a first plot with the rolling mean function
        ax = fig.add_subplot(bottom_gs[0, i])
        ax.scatter(subset["pos"],subset["gc_counts"], color="grey",alpha=0.5)
        ax.plot(subset["pos"],subset["counts_smooth"],color="red")

        ax.set_title(chr_name, rotation=30)

        ax.set_ylim(total_min, total_max)

        #Remove the axes ticks and labels
        ax.set_xticks([])
        ax.set_xlabel('')

        if(i!= 0):
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel('GC counts')

        #Create a second plot with somy annotations
        ax = fig.add_subplot(bottom_gs[1, i])


        sns.scatterplot(data=subset, x="pos", y="gc_counts", hue="somy_text", palette=custom_palette, 
                        alpha=0.7,ax=ax, legend=False)

        ax.set_ylim(total_min, total_max)

        #Remove the axes ticks and labels
        ax.set_xticks([])
        ax.set_xlabel('')

        if(i!= 0):
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel('GC counts')

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')