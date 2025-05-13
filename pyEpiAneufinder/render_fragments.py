#!/usr/bin/env python3

from collections import defaultdict
import pickle
import sys
import time
from typing import Dict, Generator, List, Tuple
import concurrent
import numpy as np
#import fargv
import gzip
from matplotlib import pyplot as plt
from skmisc.loess import loess
from scipy.ndimage import gaussian_filter1d

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
                current_processed_lines = []
        yield (cell_name, current_chr_name), current_processed_lines


def render_counts_per_window(load_fragments_by_cell_and_chr: Generator[Dict[Tuple[str, str], List[Tuple[int, int]]], None, None],
                             chr_to_windows: Dict[str, List[Tuple[int, int, float, float, float]]]) -> Generator[Tuple[Tuple[str, str], np.ndarray, np.ndarray], None, None]:
    for (cell, chromosome), frag_start_end in load_fragments_by_cell_and_chr:
        win_start_end_gc_at_n = chr_to_windows[chromosome]
        if len(win_start_end_gc_at_n) == 0:
            continue
        window_size = (win_start_end_gc_at_n[0][1] - win_start_end_gc_at_n[0][0])
        chr_len = win_start_end_gc_at_n[-1][1] + window_size * 100
        frag_starts = np.zeros(chr_len, dtype=np.int8)
        frag_ends = np.zeros(chr_len, dtype=np.int8)

        for start, end in frag_start_end:
            frag_starts[start] += 1
            frag_ends[end] += 1
            #print(".", end="")

        frag_starts = np.cumsum(frag_starts)
        frag_ends = np.cumsum(frag_ends)
        
        win_starts = np.array([x[0] for x in win_start_end_gc_at_n], dtype=np.int32) - 1
        win_ends = np.array([x[1] for x in win_start_end_gc_at_n], dtype=np.int32) - 1
        
        #print("Win:", win_starts[-1], win_ends[-1])
        #print("Frag:", frag_starts[-1], frag_ends[-1])
        gc = np.array([x[2] for x in win_start_end_gc_at_n], dtype=np.float32)
        at = np.array([x[3] for x in win_start_end_gc_at_n], dtype=np.float32)
        counts_per_window = frag_ends[win_ends] - frag_starts[win_starts]
        yield (cell, chromosome), counts_per_window, gc, at


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
        gc = np.array([x[2] for x in win_start_end_gc_at_n], dtype=np.float32)
        at = np.array([x[3] for x in win_start_end_gc_at_n], dtype=np.float32)
        avoids = np.logical_or(fragment_starts[None, :] > window_ends[:, None], fragment_ends[None, :] < window_starts[:, None]).sum(axis=1)
        counts = len(fragment_starts) - avoids        
        yield (cell, chromosome), counts, gc, at


def render_counts_per_window_gpu(load_fragments_by_cell_and_chr: Generator[Dict[Tuple[str, str], List[Tuple[int, int]]], None, None],
                             chr_to_windows: Dict[str, List[Tuple[int, int, float, float, float]]], device: str = "cuda") -> Generator[Tuple[Tuple[str, str], np.ndarray, np.ndarray], None, None]:
    import torch
    assert torch.cuda.is_available()
    for (cell, chromosome), frag_start_end in load_fragments_by_cell_and_chr:
        win_start_end_gc_at_n = chr_to_windows[chromosome]
        if len(win_start_end_gc_at_n) == 0:  # If no counts in a chromosome as in the case of chrY which is blacklisted.
            continue

        fragment_starts_ends = np.array(frag_start_end, dtype=np.int32)
        with torch.no_grad():
            fragment_starts = torch.tensor(fragment_starts_ends[:, 0], dtype=torch.int32, device=device)
            fragment_ends = torch.tensor(fragment_starts_ends[:, 1], dtype=torch.int32, device=device)

            window_starts = torch.tensor(np.array([x[0] for x in win_start_end_gc_at_n], dtype=np.int32), dtype=torch.int32, device=device)
            window_ends = torch.tensor(np.array([x[1] for x in win_start_end_gc_at_n], dtype=np.int32), dtype=torch.int32, device=device)

            #avoids = np.logical_or(fragment_starts[None, :] > window_ends[:, None], fragment_ends[None, :] < window_starts[:, None]).sum(axis=1)
            avoids = torch.logical_or(fragment_starts[None, :] > window_ends[:, None], fragment_ends[None, :] < window_starts[:, None]).sum(dim=1).detach().cpu().numpy()
        counts = len(fragment_starts) - avoids        
        gc = np.array([x[2] for x in win_start_end_gc_at_n], dtype=np.float32)
        at = np.array([x[3] for x in win_start_end_gc_at_n], dtype=np.float32)
        yield (cell, chromosome), counts, gc, at


def get_loess_smoothed(counts_per_window: np.ndarray, gc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        counts_per_window = counts_per_window
        lo = loess(gc, counts_per_window, span=0.1)
        lo.fit()
        gc_smoothed = lo.outputs.fitted_values
        return gc_smoothed
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return counts_per_window

def gaussian_convolve_1d(signal, sigma, filter_size):
    return gaussian_filter1d(signal, sigma=sigma, mode='reflect', truncate=(filter_size / (2 * sigma)))

def process_fragments(windows_csv,fragments,fragments_chunk_size, unvectorize, 
                      gpu_device, plot_each_window,pickle_counts_path):

    t = time.time()
    chr_to_windows = load_windows_dict(windows_csv)
    fragment_loader = load_fragments_by_cell_and_chr(fragments, lines_chunk=fragments_chunk_size)
    cell_chromosome_normalised = {}
    if unvectorize and gpu_device == "cpu":
        count_renderer = render_counts_per_window(fragment_loader, chr_to_windows)
    elif gpu_device != "cpu" and not unvectorize:
        count_renderer = render_counts_per_window_gpu(fragment_loader, chr_to_windows)
    elif (gpu_device == "cpu") and (not unvectorize):
        count_renderer = render_counts_per_window_vectorized(fragment_loader, chr_to_windows)
    else:
        raise ValueError("invalid combination gpu_device and unvectorized")
    
    for n, ((cell, chromosome), counts_per_window, gc, at) in enumerate(count_renderer):
        print(f"{time.time() - t:.3} {n}: {cell}\t{chromosome}: {counts_per_window.sum()}\t{gc.mean()}\t{at.mean()}", file=sys.stderr)
        #if chromosome == "chr21":
        #    print(f"couts: {counts_per_window} \n\n{counts_per_window.mean()} {counts_per_window.min()} {counts_per_window.max()}", file=sys.stderr)
        loess = get_loess_smoothed(counts_per_window, gc)
        correction = counts_per_window.mean()/(loess + .000000000001)
        loess_norm_counts = counts_per_window * correction
        if plot_each_window:
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            ax1.title.set_text('Counts per Window')
            ax2.title.set_text('Counts vs GC')
            ax3.title.set_text('Loess Nomralised Counts')
            ax4.title.set_text('Loess Counts vs GC')
            ax1.plot(counts_per_window, '.')
            ax1.plot(gaussian_convolve_1d(counts_per_window, 10, 100), 'r-')
            ax2.plot(gc, counts_per_window, '.')
            ax3.plot(loess_norm_counts, '.')
            ax3.plot(gaussian_convolve_1d(loess_norm_counts, 10, 100), 'r-')
            ax4.plot(gc, loess, '.')
            plt.suptitle(f"{chromosome} {cell} Counts:{counts_per_window.sum()} GCrange:[{gc.min()}-{gc.max()}], GCmean:{gc.mean()}")
            plt.show()
    cell_chromosome_normalised[cell, chromosome] = [counts_per_window, loess, loess_norm_counts]

    if pickle_counts_path != "":
        pickle.dump(cell_chromosome_normalised, open(pickle_counts_path, "wb"))

if __name__ == "__main__":

    windows_csv = "/work/project/ladcol_010/pyEpiAneufinder/test_run/hg38_w100000.csv"
    fragments = "/work/project/ladcol_010/epiAneufinder_improvements/epiAneufinder_test/sample_data/sample.tsv"
    #fragments = "/work/project/ladcol_010/epiAneufinder_improvements/input/SNU601/fragments.tsv.gz"
    fragments_chunk_size = 100000
    plot_each_window = False
    unvectorize =  True
    gpu_device = "cpu"
    pickle_counts_path = "/work/project/ladcol_010/pyEpiAneufinder/test_run/normalized_counts.csv"
    
    process_fragments(windows_csv,fragments,fragments_chunk_size, unvectorize, 
                      gpu_device, plot_each_window,pickle_counts_path)