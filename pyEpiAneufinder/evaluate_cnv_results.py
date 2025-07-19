import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage


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

