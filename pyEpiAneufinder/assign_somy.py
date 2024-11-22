import numpy as np

def threshold_dist_values(result_df):
    
    # Calculate the zscore (standardize the values)
    result_df['zscores'] = (result_df['ad_dist'] - result_df['ad_dist'].mean()) / result_df['ad_dist'].std()
    
    # Filter rows where zscores > 0
    result_df = result_df[result_df['zscores'] > 0]
    
    #Delete zscores again
    result_df = result_df.drop(columns=['zscores'])
    
    return result_df


def assign_gainloss(seq_data,cluster,uq=0.9,lq=0.1):
    
    #Normalize data
    counts_normal = seq_data / np.mean(seq_data)
    counts_normal[counts_normal < 0] = 0
    
    #Get global quantiles (for filtering)
    qus_global = np.quantile(seq_data, [0.01, 0.98])
    
    #Estimate trimmed mean per cluster (remove extreme quantiles before)
    grouped_data = pd.Series(counts_normal).groupby(cluster)
    cnmean = grouped_data.apply(lambda x: compute_cluster_mean(x, lq, uq, qus_global))
    
    # Identify clusters/segments with Z scores between -1 and 1
    cnmean_scaled = (cnmean - np.mean(cnmean)) / np.std(cnmean)
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
    
    qus = np.quantile(x, [lq, uq])

    # Filter values within both cluster-specific and global quantiles
    y = x[(x >= qus[0]) & (x <= qus[1]) & (x >= qus_global[0]) & (x <= qus_global[1])]

    # Fallback to the entire cluster if the filtered set is empty
    if y.sum() == 0 or len(y) == 0:
        y = x

    # Return the mean of the filtered values
    return y.mean()