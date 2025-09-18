import numpy as np
import pandas as pd

def threshold_dist_values(result_df):
    """
    Function to prune breakpoints

    Parameters
    ----------
    results_df: Pandas data frame with the AD distance of the breakpoints (column called ad_dist)

    Output
    ------
    A filtered version of the input Pandas data frame
    
    """

    # Calculate the zscore (standardize the values)
    result_df['zscores'] = (result_df['ad_dist'] - result_df['ad_dist'].mean()) / result_df['ad_dist'].std()
    
    # Filter rows where zscores > 0
    result_df = result_df[result_df['zscores'] > 0]
    
    #Delete zscores again
    result_df = result_df.drop(columns=['zscores'])
    
    return result_df


def trimmed_mean_iqr(x):
    """
    Compute mean of x after trimming outliers based on IQR rule.

    Parameters
    ----------
    x : array-like
        Numeric vector (e.g., list, NumPy array)

    Returns
    -------
    float
        Mean of values within 1.5 * IQR from Q1 and Q3
    """
    x = np.asarray(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    trimmed_x = x[(x >= lower_bound) & (x <= upper_bound)]
    return np.mean(trimmed_x)


def assign_gainloss(seq_data,cluster,uq=0.9,lq=0.1,mean_shrinking=False,trimmed_mean=False):
    """
    Function to assign CNV states

    Parameters
    ----------
    seq_data: Sequential data - Counts per bin (as a numpy array)
    cluster: Numeric list showing segment identity
    uq: Upper quantile to trim to calculate the cluster means
    lq: Lower quantile to trim to calculate the cluster means
    mean_shrinking: If True, shrink the means of the clusters with Z scores between -1 and 1 to the global mean
    trimmed_mean: If True, use the trimmed mean (based on IQR) to calculate the fold change, otherwise regular mean

    Output
    ------
    List with CNV states for each window (0=loss, 1=base, 2=gain)
    
    """
        
    #Normalize data
    seq_data[seq_data < 0] = 0    #Remove < 0 artifacts resulting from GC correction
    counts_normal = seq_data / np.mean(seq_data)
    
    #Get global quantiles (for filtering)
    qus_global = np.quantile(seq_data, [0.01, 0.98])
    
    #Estimate trimmed mean per cluster (remove extreme quantiles before)
    grouped_data = pd.Series(counts_normal).groupby(cluster)
    cnmean = grouped_data.apply(lambda x: compute_cluster_mean(x, lq, uq, qus_global))
    
    # Identify clusters/segments with Z scores between -1 and 1
    if mean_shrinking:
        cnmean_scaled = (cnmean - np.mean(cnmean)) / np.std(cnmean,ddof=1)
        cnmean_significance = (cnmean_scaled >= -1) & (cnmean_scaled <= 1)
        
        #Set these values to the mean (will become CNV status 1)
        cnmean[cnmean_significance] = np.mean(cnmean)
    
    #Calcuate the fold change
    if trimmed_mean:
        #Trimmed mean
        cnmean_fc = cnmean / trimmed_mean_iqr(cnmean)
    else:
        #Regular mean
        cnmean_fc = cnmean / np.mean(cnmean)
    
    #Truncate it to have no FC larger than 2 (all should be gain)
    cnmean_fc[cnmean_fc > 2] = 2
    
    CNV_states = round(cnmean_fc[cluster])
    return(CNV_states)


def compute_cluster_mean(x,lq,uq,qus_global):
    """
    Help function for assign_gainloss to calculate the trimmed mean

    Parameters
    ----------
    x: Counts per bin for one segment (multiple adjacent windows with same CNV status) (as a numpy array)
    uq: Upper quantile to trim to calculate the cluster means
    lq: Lower quantile to trim to calculate the cluster means
    qus_global: Quantiles of the global count distribution (list of of lower and upper quantile)

    Output
    ------
    List with CNV states for each window (0=loss, 1=base, 2=gain)
    
    """
    qus = np.quantile(x, [lq, uq])

    # Filter values within both cluster-specific and global quantiles
    y = x[(x >= qus[0]) & (x <= qus[1]) & (x >= qus_global[0]) & (x <= qus_global[1])]

    # Fallback to the entire cluster if the filtered set is empty
    if y.sum() == 0 or len(y) == 0:
        y = x

    # Return the mean of the filtered values
    return y.mean()