def threshold_dist_values(result_df):
    
    # Calculate the zscore (standardize the values)
    result_df['zscores'] = (result_df['ad_dist'] - result_df['ad_dist'].mean()) / result_df['ad_dist'].std()
    
    # Filter rows where zscores > 0
    result_df = result_df[result_df['zscores'] > 0]
    
    #Delete zscores again
    result_df = result_df.drop(columns=['zscores'])
    
    return result_df