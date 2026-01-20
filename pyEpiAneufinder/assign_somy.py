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


def trimmed_mean_iqr(x, lb=True):
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

    if lb:
        lower_bound = q1 - 1.5 * iqr
    else:
        lower_bound = 0
    upper_bound = q3 + 1.5 * iqr
    trimmed_x = x[(x >= lower_bound) & (x <= upper_bound)]
    return np.mean(trimmed_x)


def assign_gainloss(seq_data,cluster):
    """
    Function to assign CNV states

    Parameters
    ----------
    seq_data: Sequential data - Counts per bin (as a numpy array)
    cluster: Numeric list showing segment identity
        
    Output
    ------
    List with CNV states for each window (0=loss, 1=base, 2=gain)
    
    """
        
    #Normalize data
    seq_data[seq_data < 0] = 0    #Remove < 0 artifacts resulting from GC correction
    counts_normal = seq_data / np.mean(seq_data)
    
    #Estimate trimmed mean per cluster (remove extreme quantiles before)
    grouped_data = pd.Series(counts_normal).groupby(cluster)
    cnmean = grouped_data.apply(lambda x: np.mean(x))
    
    #Calcuate the fold change using trimmed mean
    cnmean_fc = cnmean / trimmed_mean_iqr(cnmean)
    
    #Truncate it to have no FC larger than 2 (all should be gain)
    cnmean_fc[cnmean_fc > 2] = 2
    
    CNV_states = round(cnmean_fc[cluster])
    return(CNV_states)


def assign_gainloss_v1(seq_data,cluster,uq=0.9,lq=0.1):
    """
    Function to assign CNV states

    Parameters
    ----------
    seq_data: Sequential data - Counts per bin (as a numpy array)
    cluster: Numeric list showing segment identity
    uq: Upper quantile to trim to calculate the cluster means
    lq: Lower quantile to trim to calculate the cluster means

    Output
    ------
    List with CNV states for each window (0=loss, 1=base, 2=gain)
    
    """
        
    #Normalize data
    counts_normal = seq_data / np.mean(seq_data)
    counts_normal[counts_normal < 0] = 0
    
    #Get global quantiles (for filtering)
    qus_global = np.quantile(seq_data, [0.01, 0.98])
    
    #Estimate trimmed mean per cluster (remove extreme quantiles before)
    grouped_data = pd.Series(counts_normal).groupby(cluster)
    cnmean = grouped_data.apply(lambda x: compute_cluster_mean(x, lq, uq, qus_global))
    
    # Identify clusters/segments with Z scores between -1 and 1
    cnmean_scaled = (cnmean - np.mean(cnmean)) / np.std(cnmean,ddof=1)
    cnmean_significance = (cnmean_scaled >= -1) & (cnmean_scaled <= 1)
    
    #Set these values to the mean (will become CNV status 1)
    cnmean[cnmean_significance] = np.mean(cnmean)
    
    #Calcuate the fold change
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


def weighted_scale_search(seg_means, seg_lengths, k_max=6, s_grid=None, median_val=None):
    """
    Weighted scale-factor grid search for CNV calling.

    Parameters
    ----------
    seg_means : array-like
        Segment means (already GC-corrected).
    seg_lengths : array-like
        Length of each segment (number of bins per segment).
    k_max : int
        Maximum copy number state to consider.
    s_grid : array-like or None
        Grid of scale factors to search over. If None, will be auto-generated.

    Output
    -------
    best_s : float
        Optimal scale factor.
    cn_states : np.ndarray
        Integer copy number state for each segment.
    scores : dict
        Mapping from scale factor -> weighted score.
    """

    # Potential check that kmax is at least 4
    #if k_max<4:
    #    raise ValueError("kmax (number of states) needs to be at least 4!")
    
    # clip seg_means at 0.5 * max integer copy number to avoid choosing too large s
    seg_means = np.asarray(seg_means)
    seg_means[seg_means > ((k_max+1)*0.5*median_val)] = (k_max+1)*0.5*median_val
    # Compute seg_lengths as numpy array
    seg_lengths = np.asarray(seg_lengths)

    if s_grid is None:
        # Generate a grid around the median segment mean
        #median_val = np.median(seg_means)
        s_grid = np.linspace(median_val * 0.5, np.min([1.2, median_val * 0.75]), 100)

    # Possible integer copy number states
    ks = np.arange(0, k_max + 1)

    # Penalize deviations from diploid (k=2)
    state_weights = np.concatenate((np.array([2.0, 1.5]),1 / np.arange(2,k_max+1)))

    # Compute scores
    scores = {}
    for s in s_grid:
        # squared distance to nearest integer * s
        d2 = ((seg_means[:, None] - s * ks[None, :]) ** 2) * state_weights[None, :]
        d2_min = d2.min(axis=1)
        score = np.sum(seg_lengths * d2_min)
        scores[s] = score

    # Pick best scale
    best_s = min(scores, key=scores.get)

    # Call resulting CN states
    # Watson: continuous
    cont_cn_states = np.round(seg_means / best_s, 2)
    cont_cn_states[cont_cn_states < 0] = 0
    cont_cn_states[cont_cn_states > k_max] = k_max

    # Holmes: discrete integer
    d2 = ((seg_means[:, None] - best_s * ks[None, :]) ** 2) * state_weights[None, :]
    int_cn_states = ks[np.argmin(d2, axis=1)]

    return best_s, cont_cn_states, int_cn_states


def assign_gainloss_new(seq_data, cluster, s_grid=None):
    """
    Function to assign CNV states

    Parameters
    ----------
    seq_data: Sequential data - Counts per bin (as a numpy array)
    cluster: Numeric list showing segment identity
    s_grid : If True, grid of scale factors to search over. If None, will be auto-generated.

    Output
    ------
    List with CNV states for each window (0=loss, 1=base, 2=gain)
    
    """
        
    #Normalize data by trimmed mean
    counts_normal = seq_data / trimmed_mean_iqr(seq_data, lb=False)
    
    #Estimate trimmed mean per cluster (remove extreme quantiles before)
    grouped_data = pd.Series(counts_normal).groupby(cluster)

    # compute segment means
    seg_means = grouped_data.apply(lambda x: np.mean(x))

    # segment lengths (# of bins per segment)
    seg_lengths = grouped_data.size()

    #Estimate the median value for the s_grid
    seg_means_per_bin = np.array([seg_means[c] for c in cluster])

    #Get the median value for the cell
    median_val = np.median(seg_means_per_bin)

    # run scale-factor grid search
    best_s, cont_cn_states, int_cn_states = weighted_scale_search(seg_means.values, seg_lengths.values, 
                                                                        s_grid=s_grid, median_val=median_val)

    # cn_states is per-segment (same order as seg_mean.index)
    # seg_mean.index contains the segment labels
    seg_labels = seg_means.index.values   # unique cluster IDs
    seg2state_cont = dict(zip(seg_labels, cont_cn_states))
    seg2state_int = dict(zip(seg_labels, int_cn_states))  # map from cluster -> CN state

    # now map each bin's cluster label to a CN state
    # Holmes
    bin_cn_int = np.array([seg2state_int[c] for c in cluster])
    bin_cn_states_holmes = np.clip(bin_cn_int, 1, 3) - 1

    # Watson
    bin_cn_cont = np.array([seg2state_cont[c] for c in cluster])
    bin_cn_states_watson = np.where(
        bin_cn_cont <= 1, 0,
        np.where(bin_cn_cont >= 3, 2, 1)
    )

    # Combined output
    bin_cn_states = np.full_like(bin_cn_states_holmes, -1.0, dtype=float)

    # exact matches
    bin_cn_states[(bin_cn_states_holmes == 0) & (bin_cn_states_watson == 0)] = 0.0
    bin_cn_states[(bin_cn_states_holmes == 1) & (bin_cn_states_watson == 1)] = 1.0
    bin_cn_states[(bin_cn_states_holmes == 2) & (bin_cn_states_watson == 2)] = 2.0

    # mixed cases
    bin_cn_states[(bin_cn_states_holmes == 0) & (bin_cn_states_watson == 1)] = 0.5
    bin_cn_states[(bin_cn_states_holmes == 2) & (bin_cn_states_watson == 1)] = 1.5

    return bin_cn_int, bin_cn_cont, bin_cn_states_holmes, bin_cn_states_watson, bin_cn_states, best_s