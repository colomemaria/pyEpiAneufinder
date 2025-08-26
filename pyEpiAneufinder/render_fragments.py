#!/usr/bin/env python3

from collections import defaultdict
import sys
from typing import Dict, Generator, List, Tuple
import numpy as np
import gzip
from skmisc.loess import loess

import pandas as pd
import natsort
import anndata as ad
from scipy.sparse import csr_matrix
import scipy.io
import os

from itertools import groupby

def autodetect_file(basepath):
    """Return path to gzipped file if available, otherwise plain file."""
    if os.path.exists(basepath + ".gz"):
        return basepath + ".gz"
    elif os.path.exists(basepath):
        return basepath
    else:
        raise FileNotFoundError(f"No file found for {basepath}(.gz)")

def read_mtx_auto(basepath):
    """Read mtx or mtx.gz file"""
    path = autodetect_file(basepath)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return scipy.io.mmread(f)
    else:
        return scipy.io.mmread(path)

def read_tsv_auto(basepath, header=None, names=None):
    """Read tsv/bed or gzipped version"""
    path = autodetect_file(basepath)
    return pd.read_csv(path, sep="\t", header=header, names=names, compression="gzip" if path.endswith(".gz") else None)

def load_windows_dict(windows_csv: str) -> Dict[str, List[Tuple[int, int, float, float, float]]]:
    with open(windows_csv) as f:
        lines = [line.split(",")[1:] for line in f.read().strip().split("\n")[1:]]

    by_chr = defaultdict(lambda: [])
    for line in lines:
        by_chr[line[0]].append((int(line[1]), int(line[2]), float(line[3]), float(line[4]), float(line[5])))
    return by_chr


def load_fragments_by_cell(fragments_path: str, lines_chunk=100000, min_frags=20000) -> Generator[Dict[str, List[Tuple[str, int, int]]], None, None]:
    """Loads a SORTED fragments file and yields fragments for each cell.
    
    The input file is sorted by cell name and then by chromosome as a numeric value (while is is stored as astring), start, and end position.
    The returned list for every cell is sorted in the same way.

    Args:
        fragments_path (str): Path to a fragments file either .tsv or .tsv.gz
        lines_chunk (int, optional): How many textlines to read at a time. Defaults to 100000.

    Yields:
        Generator[Dict[str, List[Tuple[str, int, int]]], None, None]: A generator that yields a dictionary with
            cell name as key and a sorted list of tuples with chromosome, start, and end as value.
    """
    if fragments_path.endswith(".gz"):
        f = gzip.open(fragments_path, "rt", encoding="utf-8")
    elif fragments_path.endswith(".tsv"):
        f = open(fragments_path, "r", encoding="utf-8")
    else:
        raise ValueError("Fragments file must be a .tsv or .tsv.gz file")

    unprocessed_lines = f.readlines(lines_chunk)
    current_processed_lines = []
    current_cell_name = unprocessed_lines[0].split("\t")[3]
    while unprocessed_lines:
        unprocessed_line_idx = 0
        while unprocessed_line_idx < len(unprocessed_lines):
            line = unprocessed_lines[unprocessed_line_idx].split("\t")

            if line[3] == current_cell_name:
                current_processed_lines.append((line[0], int(line[1]), int(line[2])))
                unprocessed_line_idx += 1
            else:
                if(len(current_processed_lines) >= min_frags):
                    yield current_cell_name, current_processed_lines
                current_cell_name = line[3]
                current_processed_lines = []
        unprocessed_lines = f.readlines(lines_chunk)
        
    if(len(current_processed_lines) >= min_frags):
        yield current_cell_name, current_processed_lines


def load_fragments_by_cell_and_chr(fragments_path: str, lines_chunk=100000, min_frags=20000) -> Generator[Dict[Tuple[str, str], List[Tuple[int, int]]], None, None]:
    for cell_name, all_fragments in load_fragments_by_cell(fragments_path, lines_chunk, min_frags):
        uprocessed_chr_lines = all_fragments
        current_chr_name = uprocessed_chr_lines[0][0]
        current_processed_lines = []
        for line in uprocessed_chr_lines:
            if line[0] == current_chr_name:
                current_processed_lines.append((line[1], line[2]))
            else:
                yield (cell_name, current_chr_name), current_processed_lines
                current_chr_name = line[0]
                current_processed_lines = [(line[1], line[2])]
        yield (cell_name, current_chr_name), current_processed_lines


def render_counts_per_window_vectorized(load_fragments_by_cell_and_chr: Generator[Dict[Tuple[str, str], List[Tuple[int, int]]], None, None],
                             chr_to_windows: Dict[str, List[Tuple[int, int, float, float, float]]]) -> Generator[Tuple[Tuple[str, str], np.ndarray, np.ndarray], None, None]:
    for (cell, chromosome), frag_start_end in load_fragments_by_cell_and_chr:
        win_start_end_gc_at_n = chr_to_windows[chromosome]
        if len(win_start_end_gc_at_n) == 0:  # If no counts in a chromosome as in the case of chrY which is blacklisted.
            continue
        
        fragment_starts_ends = np.array(frag_start_end, dtype=np.int32)
        fragment_starts = fragment_starts_ends[:, 0]
        fragment_ends = fragment_starts_ends[:, 1]
        
        window_starts = np.array([x[0] for x in win_start_end_gc_at_n], dtype=np.int32)
        window_ends = np.array([x[1] for x in win_start_end_gc_at_n], dtype=np.int32)
        #gc = np.array([x[2] for x in win_start_end_gc_at_n], dtype=np.float32)
        #at = np.array([x[3] for x in win_start_end_gc_at_n], dtype=np.float32)
        avoids = np.logical_or(fragment_starts[None, :] > window_ends[:, None], fragment_ends[None, :] < window_starts[:, None]).sum(axis=1)
        counts = len(fragment_starts) - avoids        
        yield (cell, chromosome), counts #, gc, at


# def render_counts_per_window_gpu(load_fragments_by_cell_and_chr: Generator[Dict[Tuple[str, str], List[Tuple[int, int]]], None, None],
#                              chr_to_windows: Dict[str, List[Tuple[int, int, float, float, float]]], device: str = "cuda") -> Generator[Tuple[Tuple[str, str], np.ndarray, np.ndarray], None, None]:
#     import torch
#     assert torch.cuda.is_available()
#     for (cell, chromosome), frag_start_end in load_fragments_by_cell_and_chr:
#         win_start_end_gc_at_n = chr_to_windows[chromosome]
#         if len(win_start_end_gc_at_n) == 0:  # If no counts in a chromosome as in the case of chrY which is blacklisted.
#             continue

#         fragment_starts_ends = np.array(frag_start_end, dtype=np.int32)
#         with torch.no_grad():
#             fragment_starts = torch.tensor(fragment_starts_ends[:, 0], dtype=torch.int32, device=device)
#             fragment_ends = torch.tensor(fragment_starts_ends[:, 1], dtype=torch.int32, device=device)

#             window_starts = torch.tensor(np.array([x[0] for x in win_start_end_gc_at_n], dtype=np.int32), dtype=torch.int32, device=device)
#             window_ends = torch.tensor(np.array([x[1] for x in win_start_end_gc_at_n], dtype=np.int32), dtype=torch.int32, device=device)

#             #avoids = np.logical_or(fragment_starts[None, :] > window_ends[:, None], fragment_ends[None, :] < window_starts[:, None]).sum(axis=1)
#             avoids = torch.logical_or(fragment_starts[None, :] > window_ends[:, None], fragment_ends[None, :] < window_starts[:, None]).sum(dim=1).detach().cpu().numpy()
#         counts = len(fragment_starts) - avoids        
#         gc = np.array([x[2] for x in win_start_end_gc_at_n], dtype=np.float32)
#         at = np.array([x[3] for x in win_start_end_gc_at_n], dtype=np.float32)
#         yield (cell, chromosome), counts, gc, at


def get_loess_smoothed(counts_per_window: np.ndarray, gc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        counts_per_window = counts_per_window
        lo = loess(gc, counts_per_window, span=0.75)
        lo.fit()
        gc_smoothed = lo.outputs.fitted_values
        return gc_smoothed
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return counts_per_window
    

def process_fragments(windows_csv,fragments,fragments_chunk_size, minFrags):

    chr_to_windows = load_windows_dict(windows_csv)


    #Get the start and length for each chromosome
    start_df = pd.DataFrame({"chromosome":chrom,"length":len(windows)}
                            for chrom, windows in chr_to_windows.items())

    #Sort alphanumerical using natsort
    start_df.sort_values(by='chromosome', key=natsort.natsort_keygen(), inplace=True)

    #Calculate consecutive start and end in the array
    start_df["start_pos"] =[0] + list(start_df["length"].cumsum()[:-1])
    start_df["end_pos"] = start_df["length"].cumsum()

    start_df.set_index("chromosome",inplace=True)
    start_pos_dict = start_df["start_pos"].to_dict()
    end_pos_dict = start_df["end_pos"].to_dict()
    
    fragment_loader = load_fragments_by_cell_and_chr(fragments, lines_chunk=fragments_chunk_size, min_frags=minFrags)
    count_renderer = render_counts_per_window_vectorized(fragment_loader, chr_to_windows)

    #Convert it to a count matrix and filter already cells with too little counts
    matrix_rows = []
    cell_ids = []

    total_length = sum(start_df["length"])
    counts_cell = np.zeros(total_length, dtype=int)

    for (cell, group) in groupby(count_renderer, key=lambda x: x[0][0]):

        counts_cell.fill(0)
        for (_, chromosome), counts_per_window in group:
            start_idx = start_pos_dict[chromosome]
            end_idx = end_pos_dict[chromosome]
            counts_cell[start_idx:end_idx] = counts_per_window
        
        matrix_rows.append(counts_cell.copy())
        cell_ids.append(cell)
    
    # Check that cells were found above the minFrags threshold
    if len(cell_ids) == 0:
        raise ValueError("No cells found! Consider reducing the parameter minFrags.")
    
    # Stack the 1D sparse arrays into a 2D sparse matrix
    matrix_2d = np.vstack(matrix_rows)
  
    # Create an AnnData object
    counts = ad.AnnData(csr_matrix(matrix_2d))
    counts.obs["cellID"] = cell_ids

    #Create one pandas data frame for the windows
    all_windows = pd.DataFrame([
        {"start": start, "end": end, "GC": GC, "AT": AT, "N": N, "chromosome": chrom}
        for chrom, windows in chr_to_windows.items()
        for start, end, GC, AT, N in windows])

    #Set metadata of Anndata frame
    counts.var["seq"]=list(all_windows["chromosome"])
    counts.var["start"]=list(all_windows["start"])
    counts.var["end"]=list(all_windows["end"])
    counts.var["GC"]=list(all_windows["GC"])

    return counts


def process_count_matrix(windows_csv, minFrags, cellRangerInput):
    """
    Process the count matrix from fragments and return an AnnData object.
    
    Args:
        windows_csv (str): Path to the CSV file containing window information.
        min_frags (int): Minimum number of fragments required for a cell to be included.
        cellRangerInput (str): Path to the folder containing the count matrix files.

    Returns:
        counts.AnnData: An AnnData object containing the processed count matrix.
    """
    
    #Loading the count matrix
    mtx_file = autodetect_file(os.path.join(cellRangerInput, "matrix.mtx"))
    barcodes_file = autodetect_file(os.path.join(cellRangerInput, "barcodes.tsv"))
    peaks_file = autodetect_file(os.path.join(cellRangerInput, "peaks.bed"))

    if not (os.path.exists(mtx_file) and os.path.exists(barcodes_file) and os.path.exists(peaks_file)):
        raise FileNotFoundError("Expected files 'matrix.mtx', 'barcodes.tsv', 'peaks.bed' not found in folder")
        
    mtx = scipy.io.mmread(mtx_file).T.tocsr()  # cells × peaks
    barcodes = pd.read_csv(barcodes_file, header=None)[0].tolist()
    peaks = pd.read_csv(peaks_file, sep="\t", header=None,names=["chromosome", "start", "end"])
    
    if len(barcodes) != mtx.shape[0]:
        raise ValueError(f"Mismatch: {len(barcodes)} barcodes vs {mtx.shape[0]} cells in matrix.")
    if peaks.shape[0] != mtx.shape[1]:
        raise ValueError(f"Mismatch: {peaks.shape[0]} peaks vs {mtx.shape[1]} features in matrix.")

    chr_to_windows = load_windows_dict(windows_csv)

    # Create global index offsets for each chromosome
    start_df = pd.DataFrame({
        "chromosome": list(chr_to_windows.keys()),
        "length": [len(windows) for windows in chr_to_windows.values()]
    })
    start_df["start_pos"] = [0] + list(start_df["length"].cumsum()[:-1])
    start_df["end_pos"] = start_df["length"].cumsum()
    start_df.set_index("chromosome", inplace=True)

    start_pos_dict = start_df["start_pos"].to_dict()

    total_length = sum(start_df["length"])

    #Mapping peaks to global indices
    peak_to_window = np.full(peaks.shape[0], -1, dtype=int)

    for chrom, windows in chr_to_windows.items():
        chrom_peaks = peaks[peaks["chromosome"] == chrom]
        if chrom_peaks.empty:
            continue

        mids = ((chrom_peaks["start"].values + chrom_peaks["end"].values) // 2)

        # Extract window boundaries
        win_starts = np.array([w[0] for w in windows])
        win_ends   = np.array([w[1] for w in windows])

        # Assign each midpoint to a window
        idx = np.searchsorted(win_ends, mids)
        idx = np.clip(idx, 0, len(win_starts) - 1) # Ensure idx is within bounds
        valid = (idx < len(win_starts)) & (mids >= win_starts[idx])

        global_offset = start_pos_dict[chrom]
        peak_to_window[chrom_peaks.index[valid]] = global_offset + idx[valid]

    # Drop unmapped peaks
    mask = peak_to_window != -1
    mtx = mtx[:, mask]
    peak_to_window = peak_to_window[mask]

    #Collapse the matrix to the windows
    data = np.ones_like(peak_to_window)
    assignment = csr_matrix(
        (data, (np.arange(len(peak_to_window)), peak_to_window)),
        shape=(len(peak_to_window), total_length)
    )

    window_mtx = mtx @ assignment  # cells × windows

    counts = ad.AnnData(window_mtx)
    counts.obs["cellID"] = barcodes

    # Build dataframe for all windows
    all_windows = pd.DataFrame([
        {"seq": chrom, "start": start, "end": end, "GC": GC}
        for chrom, windows in chr_to_windows.items()
        for start, end, GC, *rest in windows
    ])
    counts.var = all_windows.reset_index(drop=True)


    # Filter cells by total counts
    total_counts = np.array(window_mtx.sum(axis=1)).flatten()
    keep = total_counts >= minFrags
    counts = counts[keep].copy()
    if counts.n_obs == 0:
        raise ValueError(f"No cells with at least {min_frags} fragments found.")

    print(f"Final matrix: {counts.n_obs} cells × {counts.n_vars} windows")
    return counts