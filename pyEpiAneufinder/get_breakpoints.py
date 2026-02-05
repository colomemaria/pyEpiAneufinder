import numpy as np
import pandas as pd

from .distance_statistics import dist_ad, seq_dist_ad, seq_dist_ad_old

def permutation_test_ad(x, y, obs_stat, n_permutations=500, random_state=42):
    """
    Permutation test to check significance of difference between distributions 
    on the left and right side of potential breakpoint.

    Parameters
    ----------
    x: np.ndarray
        Sequential data on the left side of potential breakpoint
    y: np.ndarray
        Sequential data on the right side of potential breakpoint
    obs_stat: float
        Observed AD statistic for original segments
    n_permutations: int
        Number of permutations to perform
    random_state: int
        Random seed for reproducibility
    
    Returns
    ---------
    p-value from permutation test

    """

    # Set random seed
    rng = np.random.default_rng(random_state)
    # Save length of right segment and combine segments
    n_x = len(x)
    combined = np.concatenate([x, y])

    # Permutation test
    count = 0
    for _ in range(n_permutations):
        # Permute combined data and split at original lengths
        perm = rng.permutation(combined)
        x_perm, y_perm = perm[:n_x], perm[n_x:]
        # Calculate AD statistic for permuted data
        dist_perm = dist_ad(x_perm, y_perm)
        # Count how many times permuted stat is >= observed statistic
        if dist_perm >= obs_stat:
            count += 1

    return (count+1)/(n_permutations+1)



def global_permutation_test_ad(seq_data, len_x, len_y, observed_stat, n_permutations=500, random_state=42):
    """
    Global control: test whether the observed AD statistic between two local segments
    is stronger than expected by chance when comparing random segments anywhere in the genome.

    Parameters
    ----------
    seq_data: np.ndarray
        Sequential data for entire chromosome (counts per bin)
    len_x: int
        Length of left segment in original breakpoint test
    len_y: int
        Length of right segment in original breakpoint test
    observed_stat: float
        Observed AD statistic for original segments
    n_permutations: int
        Number of random segment pairs to sample for global test
    random_state: int
        Random seed for reproducibility
    
    Returns
    ---------
    p-value from global permutation test

    """

    # Set random seed
    rng = np.random.default_rng(random_state)
    n_total = len(seq_data)
    global_stats = []

    # Sample random pairs of segments of the same lengths as original segments and compute AD statistic
    for _ in range(n_permutations):
        # Randomly sample positions for two segments of lengths len_x and len_y
        idx = rng.choice(n_total, size=len_x + len_y, replace=False)
        x_global = seq_data[idx[:len_x]]
        y_global = seq_data[idx[len_x:]]
        global_stats.append(dist_ad(x_global, y_global))

    global_stats = np.array(global_stats)
    # Compute global p-value (fraction of random stats >= observed)
    p_global = (np.sum(global_stats >= observed_stat) + 1) / (n_permutations + 1)
    return p_global



def recursive_getbp(seq_data, seq_data_cell, k=3, depth=0, offset=0, n_permutations=500, alpha=0.05):
    """
    Recursive function to identify breakpoints in sequential data.
    Uses a permutation test to check significance before accepting breakpoints.

    Parameters
    ----------
    seq_data: np.ndarray
        Sequential data - counts per bin for a chromosome
    seq_data_cell: np.ndarray
        Sequential data - counts per bin for cell
    k: int
        Maximum recursion depth (2^k segments possible)
    depth: int
        Current recursion depth (internal use)
    offset: int
        Position offset of the current segment relative to the original sequence
    n_permutations: int
        Number of permutations for significance testing
    alpha: float
        p-value for significance test

    Returns
    -------
    DataFrame with breakpoint positions and AD distances

    """

    # Base case: stop if max depth reached or sequence too short
    if depth >= k or len(seq_data) <= 1:
        return []

    # Compute AD distances for this segment
    dist_vector = np.array(seq_dist_ad(seq_data))
    if dist_vector.size == 0:
        return []

    # Find best breakpoint in this segment
    bp_local = np.argmax(dist_vector) + 1   # Shift by 1 since otherwise we split left and not right of the left subsegment
    bp_global = bp_local + offset
    bp_dist = dist_vector[bp_local - 1] 

    # Split into left/right segments
    left_seg = seq_data[:bp_local]
    right_seg = seq_data[bp_local:]

    # Permutation test for significance
    pval = permutation_test_ad(left_seg, right_seg, bp_dist, n_permutations)
    if pval >= alpha:
        # Not significant → prune this branch
        return []
    
    # Global permutation test
    pval_global = global_permutation_test_ad(seq_data_cell, len(left_seg), len(right_seg), bp_dist, n_permutations)

    # Require both to be significant
    if pval_global >= alpha:
        return []

    # Recursive calls on left and right subsegments
    left_bps = recursive_getbp(left_seg, seq_data_cell, k=k, depth=depth+1, offset=offset, 
                               n_permutations=n_permutations, alpha=alpha)
    right_bps = recursive_getbp(right_seg, seq_data_cell, k=k, depth=depth+1, offset=offset+bp_local,
                                n_permutations=n_permutations, alpha=alpha)

    return left_bps + [(bp_global, bp_dist, pval, pval_global)] + right_bps



def recursive_getbp_df(seq_data, seq_data_cell, k=3, n_permutations=500, alpha=0.05):
    """Wrapper that returns a DataFrame instead of a list"""
    results = recursive_getbp(seq_data, seq_data_cell, k=k, n_permutations=n_permutations, alpha=alpha)
    return pd.DataFrame(results, columns=["breakpoint", "ad_dist", "p_value_local", "p_value_global"])



def getbp(seq_data, minsize=1, k=3, minsizeCNV=5):
    """
    Function to identify breakpoints (recursive research)

    Parameters
    ----------
    seq_data: array-like
        Sequential data (counts per bin)
    minsize: int
        Resolution at the level of ins. Default: 1. Setting it to higher numbers runs the algorithm faster at the cost of resolution
    k: int
        Find 2^k segments per chromosome
    minsizeCNV: int
        Number of consecutive bins to constitute a possible CNV

    Returns
    ------
    A Pandas DataFrame with all breakpoint and the AD distance at this breakpoint
    
    """
            
    # Save the position and the distance of the breakpoint
    bp = []
    dist_bp = []

    for i in range(k):
    
        # Split the sequence accordingly at the breakpoints
        seq_k_data = np.split(seq_data,sorted(bp))

        total_position = 0
        for segement in seq_k_data:
        
            # Process only segements with length > 1
            if(len(segement)>1):
                # Calculate the AD distance separately for each breakpoint and identify the maximum
                dist_vector = seq_dist_ad_old(segement,minsize=minsize)
                
                # Get the position of the maximum AD distance (in the total vector seq_data)
                bp_pos_shifted = ((np.argmax(dist_vector)+1)*minsize)-1
                
                # Because of the minsize the shift might sometimes be outside the segment length 
                #(set it to the length in the this case)
                if bp_pos_shifted >= len(segement):
                    bp_pos_shifted = len(segement) - 1
                
                # Check whether it is overlapping with any other segment (works only if bp is not empty)
                if bp:
                    bp_neighbors = np.concatenate([np.arange(x - minsizeCNV, x + minsizeCNV + 1) for x in bp])
                    if not ((bp_pos_shifted+total_position) in  bp_neighbors):
                        bp.append(bp_pos_shifted + total_position)
                        # Save also the maximum AD distance
                        dist_bp.append(max(dist_vector))
                else:
                    bp.append(bp_pos_shifted + total_position)
                    # Save also the maximum AD distance
                    dist_bp.append(max(dist_vector))


            # Add the length of this segement as total counter
            total_position += len(segement)
            
    # Remove breakpoints at the beginning or the end of the segement that are too short
    bp_filtered = []
    dist_bp_filtered = []
    # Make this distinction to be exactly the same as in the R version of epiAneufinder
    if minsizeCNV > 0: 
        for i in range(len(bp)):
            if (bp[i] > 0 + minsizeCNV-1) & (bp[i]< (len(seq_data)-minsizeCNV-1)) :
                bp_filtered.append(bp[i])
                dist_bp_filtered.append(dist_bp[i])
    else:
        for i in range(len(bp)):
            if (bp[i] > 0) & (bp[i]< (len(seq_data)-1)) :
                bp_filtered.append(bp[i])
                dist_bp_filtered.append(dist_bp[i])
 
    return(pd.DataFrame({"breakpoint": np.array(bp_filtered, dtype="int64"),
                         "ad_dist": np.array(dist_bp_filtered, dtype="float64")}))



def fast_getbp(seq_data, minsize=1, k=3, minsizeCNV=5):
    """
    Function to identify breakpoints (recursive research) with a faster implementation

    Parameters
    ----------
    seq_data: array-like
        Sequential data (counts per bin)
    minsize: int
        Resolution at the level of ins. Default: 1. Setting it to higher numbers runs the algorithm faster at the cost of resolution
    k: int
        Find 2^k segments per chromosome
    minsizeCNV: int
        Number of consecutive bins to constitute a possible CNV

    Returns
    ------
    A Pandas DataFrame with all breakpoint and the AD distance at this breakpoint
    
    """

    # Save the position and the distance of the breakpoint
    bp = []
    dist_bp = []

    for _ in range(k):
       
        segments = np.split(seq_data, sorted(bp))  # Segment the sequence at current breakpoints

        # Compute start index of each segment in original array
        seg_starts = np.cumsum([0] + [len(seg) for seg in segments[:-1]]) #

        for segment, seg_start in zip(segments, seg_starts):
            if len(segment) <= 1:
                continue

            # Compute AD distances for current segment
            dist_vector = np.array(seq_dist_ad_old(segment, minsize=minsize))
            if dist_vector.size == 0:
                continue
            # Get index of max AD distance
            bp_local = ((np.argmax(dist_vector) + 1) * minsize) - 1
            bp_local = min(bp_local, len(segment) - 1)  # using min to avoiddoing it manually
            bp_global = bp_local + seg_start

            # Enforce minsizeCNV spacing from other breakpoints
            if bp:
                # OLD: bp_neighbors = np.concatenate([np.arange(x - minsizeCNV, x + minsizeCNV + 1) for x in bp])
                neighbors = set(np.concatenate([np.arange(b - minsizeCNV, b + minsizeCNV + 1) for b in bp]))
                if bp_global not in neighbors:
                    bp.append(bp_global)
                    dist_bp.append(np.max(dist_vector))
            else:
                bp.append(bp_global)
                dist_bp.append(np.max(dist_vector))
            
    bp_arr = np.array(bp, dtype=int)
    dist_arr = np.array(dist_bp, dtype=float)

    if minsizeCNV > 0:
        valid_mask = (bp_arr >= minsizeCNV) & (bp_arr <= len(seq_data) - minsizeCNV - 1)
    else:
        valid_mask = (bp_arr >= 1) & (bp_arr <= len(seq_data) - 2)

    return pd.DataFrame({"breakpoint": bp_arr[valid_mask],"ad_dist": dist_arr[valid_mask]})
    