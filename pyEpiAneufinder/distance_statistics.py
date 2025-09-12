import numpy as np

def dist_ad(x, y):
    """
    Function to calculate the AD statistic between two distributions

    Parameters
    ----------
    x: Numeric list with reads counts left of the breakpoint
    y: Numeric list with reads counts right of the breakpoint

    Output
    ------
    Anderson-darling distance

    """

    # Calculate lengths
    n = len(x)
    m = len(y)
    poolsize = n + m

    # Distinct values and their counts
    pooldistinct, counts = np.unique(np.concatenate([x, y]), return_counts=True)

    # Remove the last element (don't iterate over this one)
    pooldistinct = pooldistinct[:-1]
    counts = counts[:-1]
    
    # Calculate cumulative sums (CDFs) for x, y, and pooled
    cdf_x = np.searchsorted(np.sort(x), pooldistinct, side='right')
    cdf_y = np.searchsorted(np.sort(y), pooldistinct, side='right')
    bj = cdf_x + cdf_y

    num_x = counts * ((poolsize * cdf_x - n * bj) ** 2)
    num_y = counts * ((poolsize * cdf_y - m * bj) ** 2)
    denom = poolsize * bj * (poolsize - bj)

    sum_x = np.sum(num_x / denom)
    sum_y = np.sum(num_y / denom)

    # Final statistic
    stat_ad = (sum_x / n) + (sum_y / m)
    return stat_ad



def seq_dist_ad(seq_data):
    """
    Function to calculate the breakpoints with AD stat given a series of data points

    Parameters
    ----------
    seq_data: Normalized read counts per window (as a numeric list)
    
    Output
    ------
    List with distances for each potential breakpoint
    
    """
    
    # Create the list of breakpoints to test (with a stepsize of minsize)
    bp1 = np.arange(0, len(seq_data))

    # Loop over break points
    n_bps=len(bp1) - 1
    distlist = np.empty(n_bps)
    for i in range(n_bps):
        # Call dist_ad function
        # In the old version, the central bin was shared by both the left and right segment
        # Now, the breakpoint is located inbetween the segments
        distlist[i] = dist_ad(seq_data[:(bp1[i+1])], seq_data[bp1[i+1]:])
    
    # Replace NaN values with 0 in distlist
    distlist = np.nan_to_num(distlist)
    
    return list(distlist)



def seq_dist_ad_old(seq_data, minsize=1):
    """
    Function to calculate the breakpoints with AD stat given a series of data points

    Parameters
    ----------
    seq_data: Normalized read counts per window (as a numeric list)
    minsize: Integer. Resolution at the level of ins. Default: 1. Setting it to higher numbers runs the algorithm faster at the cost of resolution

    Output
    ------
    List with distances for each potential breakpoint
    
    """
    
    #Create the list of breakpoints to test (with a stepsize of minsize)
    bp1 = np.arange(0, len(seq_data), minsize)

    # Loop over break points
    n_bps=len(bp1)
    distlist = np.empty(n_bps)
    for i in range(n_bps):
        
        # Call dist_ad function (the middle element is taken twice to match the R code)
        distlist[i] = dist_ad(seq_data[:(bp1[i]+1)], seq_data[bp1[i]:])
    
    # Replace NaN values with 0 in distlist
    distlist = np.nan_to_num(distlist)
    
    return list(distlist)
