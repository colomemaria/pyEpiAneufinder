import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

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
    Pandas data frame with two columns of barcode and subclone group
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