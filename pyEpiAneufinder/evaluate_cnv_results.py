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