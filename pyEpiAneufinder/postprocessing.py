import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import mode

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def split_subclones_distance(res, dist_cutoff, cluster_method="complete"):
    """
    Alternative function to split the dataset into subclones, based on distance

    Parameters
    ----------
    res: Results from pyEpiAneufinder main function (as pandas data frame)
    dist_cutoff: Maximum distance inside the cluster

    Output
    ------
    Pandas data frame with two columns of barcode and subclone group
    """

    #Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq","start","end"])

    #Calculate pairwise distances between cells
    dist_matrix = pdist(data_matrix.T,metric="cityblock")

    #Normalize the distance by the number of bins (=> mean absolute error) 
    fract_deviation = dist_matrix / res.shape[0]

    #Calcuate the hierarchical clustering based on these
    hc_cluster = linkage(fract_deviation, method=cluster_method)

    #Split into groups
    cl_members = fcluster(Z=hc_cluster, t=dist_cutoff, criterion='distance')
        
    clones = pd.DataFrame({"barcode":data_matrix.columns.values,
                        "subclone":cl_members})

    return clones




def cnv_imputation_subclones(res, dist_cutoff,
                            min_clone_size=10, frac_min_occ=0.9):
    
    """
    Imputation strategy to reduce noise of CNV calls by replacing CNV status
    of a cell within a well-defined subclone of certain size with majority

    Parameters
    ----------
    res: Results from pyEpiAneufinder main function (as pandas data frame)
    dist_cutoff: Maximum distance inside the cluster (i.e. subclone)
    min_clone_size: Minimum number of cells for a cluster to run imputation on (otherwise skip)
    frac_min_occ: Minimum fraction of cells within a cluster that have the same CNV status to impute it for all cells

    Output
    ------
    Pandas data frame with two columns of barcode and subclone group
    """

    #Get clone information
    clones = split_subclones_distance(res,dist_cutoff)

    #Remove position information (only CNVs kept)
    data_matrix = res.drop(columns=["seq","start","end"])
    dm_numpy = data_matrix.to_numpy()

    #Estimate for each clone a profile of x% concordance:
    counts_imputed = np.full(dm_numpy.shape, np.nan)

    for clone in clones.subclone.unique():
        #Extract all members
        subset = dm_numpy[:,clones.subclone==clone]
        
        #Min occurance
        min_occ = subset.shape[1] * frac_min_occ
        
        #Clean only profile for at least 5 cells
        if(subset.shape[1]>=min_clone_size):
            #Get most frequent element for each bin
            modes, counts = mode(subset, axis=1)
            
            # Flatte
            modes = modes.ravel()
            counts = counts.ravel()
            
            #Replace the one which are below the threshold
            mask = counts > min_occ
            subset[mask, :] = np.repeat(modes[mask, np.newaxis], subset.shape[1], axis=1)
            
        counts_imputed[:,clones.subclone==clone] = subset

    #Convert it back to Pandas Data Frame and save it again
    imputed_pd = pd.DataFrame(counts_imputed)
    imputed_pd.index = res.index
    imputed_pd = pd.concat([res.loc[:,["seq","start","end"]],imputed_pd],axis=1)
    imputed_pd.columns = res.columns.values
    return imputed_pd
