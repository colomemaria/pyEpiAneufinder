import pandas as pd
import numpy as np
from natsort import natsorted

def compute_cnv_burden_cell(df,offset=3):
    """
    Function to compute the CNV burden (=aneuploidy score) per cell

    Parameters
    ----------
    df: Results from pyEpiAneufinder main function (as pandas data frame)
    offset: Number of columns to skip at the beginning (default the three annotation columns)

    Output
    ------
    Pandas data frame with two columns of barcodes and cnv_burden
    
    """
    cn_matrix = df.iloc[:, offset:].to_numpy()
    cnv_burden = pd.DataFrame({"barcodes": df.columns.values[offset:],
                "cnv_burden":np.mean(cn_matrix != 1, axis=0)})
    return cnv_burden

def compute_aneuploidy_across_sample(df, offset=3):
    """
    Function to compute one aneuploidy score for the complete dataset

    Parameters
    ----------
    df: Results from pyEpiAneufinder main function (as pandas data frame)
    offset: Number of columns to skip at the beginning (default the three annotation columns)

    Output
    ------
    Aneuploidy score (one numeric value)
    
    """
    cn_matrix = df.iloc[:, offset:].to_numpy()
    return np.mean(cn_matrix != 1)

def compute_aneuploidy_by_chr(df, offset=3):
    """
    Function to compute one aneuploidy score per chromosome

    Parameters
    ----------
    df: Results from pyEpiAneufinder main function (as pandas data frame)
    offset: Number of columns to skip at the beginning (default the three annotation columns)

    Output
    ------
    Pandas DataFrame with one aneuploidy score per chromosome
    
    """
    cn_matrix = df.iloc[:, offset:].to_numpy()
    chroms = df['seq'].values
    unique_chroms = natsorted(np.unique(chroms))

    result = {}
    for chr_ in unique_chroms:
        idx = np.where(chroms == chr_)[0]
        result[chr_] = np.mean(cn_matrix[idx] != 1)

    df_result = pd.DataFrame(result, index=[0])  # index is just zero
    return df_result

#Help function for compute_heterogeneity_across_sample()
def compute_heterogeneity_array(arr):
    heterogeneity = np.zeros(arr.shape[0])
    for i, row in enumerate(arr):
        vals, counts = np.unique(row, return_counts=True)
        counts = np.sort(counts)[::-1]
        weights = np.arange(len(counts))
        heterogeneity[i] = np.sum(counts * weights) / arr.shape[1]
    return heterogeneity

def compute_heterogeneity_across_sample(df, offset=3):
    """
    Function to compute one heterogeneity score for the complete dataset

    Parameters
    ----------
    df: Results from pyEpiAneufinder main function (as pandas data frame)
    offset: Number of columns to skip at the beginning (default the three annotation columns)

    Output
    ------
    Heterogeneity score (one numeric value)
    
    """
    cn_matrix = df.iloc[:, offset:].to_numpy()
    heterogeneity = compute_heterogeneity_array(cn_matrix)
    return np.mean(heterogeneity)

def compute_heterogeneity_by_chr(df, offset=3):
    """
    Function to compute one heterogeneity score per chromosome

    Parameters
    ----------
    df: Results from pyEpiAneufinder main function (as pandas data frame)
    offset: Number of columns to skip at the beginning (default the three annotation columns)

    Output
    ------
    Pandas DataFrame with one heterogeneity score per chromosome
    
    """
    cn_matrix = df.iloc[:, offset:].to_numpy()
    chroms = df['seq'].values
    unique_chroms = natsorted(np.unique(chroms))

    result = {}
    for chr_ in unique_chroms:
        idx = np.where(chroms == chr_)[0]
        heterogeneity = compute_heterogeneity_array(cn_matrix[idx])
        result[chr_] = np.mean(heterogeneity)

    df_result = pd.DataFrame(result, index=[0])  # index is just zero
    return df_result


