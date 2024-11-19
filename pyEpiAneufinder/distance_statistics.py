import numpy as np

# Function to calculate the AD statistic between two distributions
def dist_ad(x,y):
    # Calculate lengths
    n = len(x)
    m = len(y)
    poolsize = n + m

    # Pool of distinct values
    poolvec = np.concatenate([x, y])
    pooldistinct = np.unique(np.sort(poolvec))

    # Initialize sums
    sum_x = 0
    sum_y = 0

    # Iterate over distinct values in the pool
    for j in range(len(pooldistinct) - 1):
        lj = np.sum(x == pooldistinct[j]) + np.sum(y == pooldistinct[j])
        mxj = np.sum(x <= pooldistinct[j])
        myj = np.sum(y <= pooldistinct[j])
        bj = np.sum(x <= pooldistinct[j]) + np.sum(y <= pooldistinct[j])

        denom = poolsize * (bj * (poolsize - bj))
        num_x = lj * ((poolsize * mxj - n * bj) ** 2)
        num_y = lj * ((poolsize * myj - m * bj) ** 2)

        sum_x += num_x / denom
        sum_y += num_y / denom

    # Final statistic
    stat_ad = (sum_x / n) + (sum_y / m)
    return(stat_ad)

# Function to calculate the breakpoints with AD stat given a series of data points
def seq_dist_ad(seq_data,minsize=3):
    
    #Create the list of breakpoints to test (with a stepsize of minsize)
    bp1 = np.arange(0, len(seq_data), minsize)
    
    # Loop over break points
    distlist = []
    for i1 in range(len(bp1)):
        
        # Call dist_ad function
        dist_value = dist_ad(seq_data[:(bp1[i1]+1)], seq_data[bp1[i1]:])
        distlist.append(dist_value)  # Append the result to distlist
    
    # Replace NaN values with 0 in distlist
    distlist = [np.nan_to_num(d) for d in distlist]
    
    return distlist
