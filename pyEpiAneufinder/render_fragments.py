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

#only for debuggin purpose
import time

def load_windows_dict(windows_csv: str) -> Dict[str, List[Tuple[int, int, float, float, float]]]:
    with open(windows_csv) as f:
        lines = [line.split(",")[1:] for line in f.read().strip().split("\n")[1:]]

    by_chr = defaultdict(lambda: [])
    for line in lines:
        by_chr[line[0]].append((int(line[1]), int(line[2]), float(line[3]), float(line[4]), float(line[5])))
    return by_chr


def load_fragments_by_cell(fragments_path: str, lines_chunk=100000) -> Generator[Dict[str, List[Tuple[str, int, int]]], None, None]:
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
                yield current_cell_name, current_processed_lines
                current_cell_name = line[3]
                current_processed_lines = []
        unprocessed_lines = f.readlines(lines_chunk)
    yield current_cell_name, current_processed_lines


def load_fragments_by_cell_and_chr(fragments_path: str, lines_chunk=100000) -> Generator[Dict[Tuple[str, str], List[Tuple[int, int]]], None, None]:
    for cell_name, all_fragments in load_fragments_by_cell(fragments_path, lines_chunk):
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
    start_tmp=[]
    for chrom in chr_to_windows:
        start_tmp.append({"chromosome":chrom,"length":len(chr_to_windows[chrom])})
        
    start_df = pd.DataFrame(start_tmp)

    #Sort alphanumerical using natsort
    start_df.sort_values(by='chromosome', key=natsort.natsort_keygen(), inplace=True)

    #Calculate consecutive start and end in the array
    start_df["start_pos"] =[0] + list(start_df["length"].cumsum()[:-1])
    start_df["end_pos"] = start_df["length"].cumsum()

    start_df.set_index("chromosome",inplace=True)
    start_pos_dict = start_df["start_pos"].to_dict()
    end_pos_dict = start_df["end_pos"].to_dict()
    
    fragment_loader = load_fragments_by_cell_and_chr(fragments, lines_chunk=fragments_chunk_size)
    count_renderer =render_counts_per_window_vectorized(fragment_loader, chr_to_windows)

    #Convert it to a count matrix and filter already cells with too little counts
    current_cell = None
    matrix_rows = []
    cell_ids = []

    total_length = sum(start_df["length"])
    counts_cell = np.zeros(total_length, dtype=int)

    #DEBUG:
    start = time.perf_counter()

    for (cell, chromosome), counts_per_window in count_renderer:
        
        #print(cell+" "+chromosome)
        
        if cell != current_cell:
            if counts_cell.sum() > minFrags:
                matrix_rows.append(counts_cell.copy())
                cell_ids.append(current_cell)
            counts_cell.fill(0)
            current_cell = cell
            current_time=time.perf_counter()-start
            print(f"{current_cell} - total time {current_time:.1f} s")
            
        counts_cell[start_pos_dict[chromosome]:end_pos_dict[chromosome]] = counts_per_window
    
    #Save the last cells counts
    if counts_cell.sum() > minFrags:
        matrix_rows.append(counts_cell.copy())
        cell_ids.append(current_cell)

    current_time=time.perf_counter()-start
    print(f"start constructing the matrix - total time {current_time:.1f} s")    
    # Stack the 1D sparse arrays into a 2D sparse matrix
    matrix_2d = np.vstack(matrix_rows)

    print(f"start creating the anndata object - total time {current_time:.1f} s")  
    # Create an AnnData object
    counts = ad.AnnData(csr_matrix(matrix_2d))
    counts.obs["cellID"] = cell_ids

    #Create one pandas data frame for the windows
    all_windows = pd.DataFrame()
    for chrom in chr_to_windows.keys():
        tmp = pd.DataFrame(chr_to_windows[chrom], columns = ["start","end","GC","AT","N"])
        tmp["chromosome"] = chrom
        all_windows = pd.concat([all_windows,tmp],ignore_index=True)

    #Set metadata of Anndata frame
    counts.var["seq"]=list(all_windows["chromosome"])
    counts.var["start"]=list(all_windows["start"])
    counts.var["end"]=list(all_windows["end"])
    counts.var["GC"]=list(all_windows["GC"])

    return counts