import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import seaborn as sns


def karyo_gainloss(res, outdir, title, state_type='categorical', n_states=5, 
                   linkage_method='ward', dist_metric='euclidean'):
    """
    Function to plot the final karyogram with support for categorical, integer, or continuous CNV states

    Parameters
    ----------
    res: Pandas data frame with position information in the first three columns (seq, start, end), 
         followed by the CNV status of each cell
    outdir: Path to saved karyogram image
    title: Plot title
    state_type: Type of states to plot. Options: 'categorical', 'integer', 'continuous'
    n_states: Number of categorical states (3 or 5). Only used when state_type='categorical'
    linkage_method: Linkage method for hierarchical clustering (default: 'ward')
    dist_metric: Distance metric for clustering (default: 'euclidean')

    Output
    ------
    A filtered version of the input Pandas data frame
    
    """
    # Validate state_type parameter
    valid_state_types = ['categorical', 'integer', 'continuous']
    if state_type not in valid_state_types:
        raise ValueError(f"state_type must be one of {valid_state_types}")
    
    # Check required columns
    required_columns = ['seq', 'start', 'end']
    for col in required_columns:
        if col not in res.columns:
            raise KeyError(f"Missing required column in DataFrame: {col}")
        
    # Check if the DataFrame is empty
    if res.shape[0] == 0:
        raise ValueError("Input DataFrame is empty")

    # Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq", "start", "end"])

    # Check that the dataframe is not empty after dropping
    if data_matrix.shape[1] == 0 or data_matrix.shape[0] == 0:
        raise ValueError("Data matrix is empty after dropping position columns")

    # Check that all CNV values are numeric
    if not all(pd.api.types.is_numeric_dtype(data_matrix[col]) for col in data_matrix.columns):
        raise ValueError("All CNV columns must be numeric")
    
    # Validate data based on state_type
    if state_type == 'categorical':
        if n_states not in [3, 5]:
            raise ValueError("n_states must be 3 or 5 for categorical state_type")
        # For categorical, check if we have the expected discrete values
        if not ((data_matrix == 0) | (data_matrix == 1) | (data_matrix == 2)).any().any():
            raise ValueError("No gain or loss detected in the CNV data")
    
    # Calculate pairwise distances between cells
    dist_matrix = pdist(data_matrix.T, metric=dist_metric)

    # Hierarchical clustering
    Z = linkage(dist_matrix, method=linkage_method)

    # Get the relative percentage of entries per chromosome
    chr_num_bins = res["seq"].value_counts(sort=False)
    perc_chr = [chr / sum(chr_num_bins) for chr in chr_num_bins]

    # Make sure the chromosomes are sorted in the correct order in the plots
    res['seq'] = pd.Categorical(res['seq'], categories=chr_num_bins.index, ordered=True)

    # Create a figure with a custom GridSpec layout
    fig = plt.figure(figsize=(22, 8))
    gs = gridspec.GridSpec(1, len(perc_chr)+1, width_ratios=[2]+[20 * p for p in perc_chr])

    # Add the dendrogram as the first column
    ax = fig.add_subplot(gs[0, 0])
    dendro = dendrogram(Z, orientation="left", link_color_func=lambda k: 'darkgrey', ax=ax)
    # Remove the axes
    ax.axis('off')

    # Reorder the data matrix correctly
    leaf_order = dendro['leaves']  # Extract the order of the leaves as a list
    leaf_order = [l + 3 for l in leaf_order[::-1]]  # Reverse the order and add + 3 (first three columns)
    res = res.iloc[:, [0, 1, 2] + leaf_order]

    # Define colors, colorbar, and legend based on state_type
    if state_type == 'categorical':
        if n_states == 3:
            cmap = ["#9A32CD", "#00EE76", "#CD0000"]  # Loss, Base, Gain
            legend_elements = [
                Patch(facecolor="#9A32CD", edgecolor='black', label='Loss'),
                Patch(facecolor="#00EE76", edgecolor='black', label='Base'),
                Patch(facecolor="#CD0000", edgecolor='black', label='Gain')
            ]
        elif n_states == 5:
            cmap = ["#9A32CD", "#D7A0E8", "#00EE76", "#F08080", "#CD0000"]
            legend_elements = [
                Patch(facecolor="#9A32CD", edgecolor='black', label='Loss'),
                Patch(facecolor="#D7A0E8", edgecolor='black', label='Putative Loss'),
                Patch(facecolor="#00EE76", edgecolor='black', label='Base'),
                Patch(facecolor="#F08080", edgecolor='black', label='Putative Gain'),
                Patch(facecolor="#CD0000", edgecolor='black', label='Gain')
            ]
        vmin, vmax = 0, 2
        show_cbar = False
        legend_elements_to_show = legend_elements
        
    elif state_type in ['integer', 'continuous']:
        # Blue-white-red colormap centered at 2
        base_cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)
        shifted_cmap = shiftedColorMap(
            base_cmap,
            midpoint=1/3,
            name='bwr_shifted_2'
        )
        show_cbar = True
        legend_elements_to_show = None
        cbar_label = 'Integer state' if state_type == 'integer' else 'Continuous score'

    # Plot per chromosome
    chromosome_groups = list(res.groupby('seq', observed=True))
    n_chromosomes = len(chromosome_groups)

    for i, (seq, group) in enumerate(chromosome_groups):
        ax = fig.add_subplot(gs[0, i+1])
        data_filtered = group.drop(columns=["seq", "start", "end"])
        
        # Create heatmap
        if state_type == 'categorical':
            sns.heatmap(data_filtered.T, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
        else:
            # For integer and continuous, use a diverging colormap with colorbar
            sns.heatmap(data_filtered.T, ax=ax, cmap=shifted_cmap,
                        vmin=0, vmax=6, 
                        cbar=show_cbar and i == n_chromosomes-1,
                        cbar_kws={'label': cbar_label, 'pad': 0.5, 
                                  'fraction': 0.25, 'aspect': 30,
                                  'spacing': 'proportional'})
            
            # Customize the colorbar if it was just created
            if show_cbar and i == n_chromosomes - 1:
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([0, 2, 6])
                cbar.set_ticklabels(['0', '2', '6'])
        
        ax.set_title(seq, rotation=30)
        ax.set_xticks([])
        ax.set_yticks([])
     
    # Add a common x-axis label
    fig.text(0.5, 0.0, 'Position in chromosome', ha='center', va='center', fontsize=12)

    # Add legend for categorical states
    if state_type == 'categorical':
        fig.legend(handles=legend_elements_to_show, loc='lower center', 
                  ncol=len(legend_elements_to_show), fontsize=12, frameon=False,
                  bbox_to_anchor=(0.5, -0.08))
        plt.subplots_adjust(bottom=0.2)
    else:
        # For numeric states, adjust for colorbar
        plt.subplots_adjust(bottom=0.15, right=0.95)

    # Show the plot
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outdir, dpi=300, bbox_inches='tight')

    return res


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. 
    
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
      
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
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