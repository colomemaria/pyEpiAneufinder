import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns


def karyo_gainloss (res, outdir, title_karyo, n_colors=5):
    """
    Function to plot the final karyogram

    Parameters
    ----------
    res: Pandas data frame with position information in the first three columns (seq, start, end), followed by the CNV status of each cell
    outdir: Path to save the karyogram image
    title_karyo: Option to give a plot title

    Output
    ------
    A filtered version of the input Pandas data frame
    
    """
    #Check required columns
    required_columns = ['seq', 'start', 'end']
    for col in required_columns:
        if col not in res.columns:
            raise KeyError(f"Missing required columns in DataFrame: {col}")
        
    #Check if the DataFrame is empty
    if res.shape[0] == 0:
        raise ValueError("Input DataFrame is empty")

    #Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq","start","end"])

    # Check that the dataframe is not empty after dropping
    if data_matrix.shape[1] == 0 or data_matrix.shape[0] == 0:
        raise ValueError("Data matrix is empty after dropping position columns")

    # Check that all CNV values are numeric
    if not all(pd.api.types.is_numeric_dtype(data_matrix[col]) for col in data_matrix.columns):
        raise ValueError("All CNV columns must be numeric")
    
    # data_matrix already contains only CNV columns
    if not ((data_matrix == 0) | (data_matrix == 2)).any().any():
        raise ValueError("No gain or loss detected in the CNV data")

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
    gs = gridspec.GridSpec(1, len(perc_chr)+1, width_ratios=[2]+[20 * p for p in perc_chr])

    #Add the dendogram as the first column
    ax = fig.add_subplot(gs[0, 0])
    dendro = dendrogram(Z,orientation="left", link_color_func=lambda k: 'darkgrey',ax=ax)
    #Remove the axes
    ax.axis('off')

    #Reorder the data matrix correctly
    leaf_order = dendro['leaves']  # Extract the order of the leaves as a list
    leaf_order = [l + 3 for l in leaf_order[::-1]] # Reverse the order and add + 3 (first three columns)
    res = res.iloc[:, [0,1,2]+leaf_order]

        # Define colors and legend based on n_colors
    if n_colors == 3:
        cmap = ["#9A32CD", "#00EE76", "#CD0000"]  # Loss, Base, Gain
        legend_elements = [
            Patch(facecolor="#9A32CD", edgecolor='black', label='Loss'),
            Patch(facecolor="#00EE76", edgecolor='black', label='Base'),
            Patch(facecolor="#CD0000", edgecolor='black', label='Gain')
        ]
    elif n_colors == 5:
        cmap = ["#9A32CD", "#D7A0E8", "#00EE76", "#F08080", "#CD0000"]  # Loss, Putative Loss, Base, Putative Gain, Gain
        legend_elements = [
            Patch(facecolor="#9A32CD", edgecolor='black', label='Loss'),
            Patch(facecolor="#D7A0E8", edgecolor='black', label='Putative Loss'),
            Patch(facecolor="#00EE76", edgecolor='black', label='Base'),
            Patch(facecolor="#F08080", edgecolor='black', label='Putative Gain'),
            Patch(facecolor="#CD0000", edgecolor='black', label='Gain')
        ]
    else:
        raise ValueError("n_colors must be 3 or 5")

    # Plot per chromosome
    for i, (seq, group) in enumerate(res.groupby('seq', observed=True)):
        ax = fig.add_subplot(gs[0, i+1])
        data_filtered = group.drop(columns=["seq", "start", "end"])
        sns.heatmap(data_filtered.T, ax=ax, cmap=cmap, vmin=0, vmax=2, cbar=False)
        ax.set_title(seq, rotation=30)
        ax.set_xticks([])
        ax.set_yticks([])
     
    # Add a common x-axis label
    fig.text(0.5, 0.0, 'Position in chromosome', ha='center', va='center', fontsize=12)

    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    # Adjust layout
    plt.subplots_adjust(bottom=0.2)

    # Show the plot
    plt.suptitle(title_karyo)
    plt.tight_layout()
    plt.savefig(outdir+"Karyogram.png", dpi=300, bbox_inches='tight')

    return res
