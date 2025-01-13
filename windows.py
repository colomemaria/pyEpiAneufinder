#!/usr/bin/env python3

from collections import defaultdict
import sys
from typing import Dict, Generator, List, Tuple, Union
from Bio import SeqIO
from Bio.SeqUtils import nt_search
import fargv
import pandas as pd
import time
import numpy as np
import tqdm


# written by anguelos
def fasta_loader_highmem(fasta_path: str) -> Generator[Tuple[str, str], None, None]:  # Load the genome from a FASTA file
    """Fast fasta file loader

    Lads the whole fasta file into memory and then yields the chromosome name and sequence.

    Runs in 5.5 sec. vs. 14.5 sec. for SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

    Args:
        fasta_path (str): path to a fasta file

    Yields:
        Generator[Tuple[str, str], None, None]: A generator that yields tuples of chromosome name and sequence
    """
    with open(fasta_path, "r") as f:
        buffer = f.read()
    if not buffer:
        return

    start = 1
    while True:
        index = buffer.find(">", start)
        if index == -1:  # No more delimiters found
            index = len(buffer)
        name_end_idx = buffer.find("\n", start)
        chromosome_name = buffer[start:name_end_idx]
        chromosome_seq = buffer[name_end_idx:index].replace("\n", "")
        yield chromosome_name, chromosome_seq
        if index >= len(buffer):  # No more delimiters found
            break
        start = index + len(">")


def read_bed_file(bed_path: str) -> pd.DataFrame: # Read the BED file into a DataFrame
    bed_df = pd.read_csv(bed_path, sep='\t', header=None, names=['chromosome', 'start', 'end'],index_col=False)
    return bed_df


def read_bedfile_chr_dict(bed_path: str) -> Dict[str, List[Tuple[int, int, str]]]:  # Read the BED file into a dictionary
    bed_dict = defaultdict(lambda: [])
    with open(bed_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.split("\t")
            if len(line) < 4:
                print(f"Error: {line}", file=sys.stderr)
                continue
            bed_dict[line[0]].append((int(line[1]), int(line[2]), line[3]))
    return {k: sorted(v) for k, v in bed_dict.items()}


def make_windows(genome_path : str, blacklisted_bed_path :str, window_size: int, exclude: Union[List[str], None] = None) -> pd.DataFrame:
    # Load the genome from a FASTA file
    t = time.time()
    print(f"{time.time() -t:.3} Loading genome file", file=sys.stderr)
    genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
    

    # Read the blacklist BED file
    print(f"{time.time() -t:.3}  Loading blacklist file", file=sys.stderr)
    blacklist_df = read_bed_file(blacklisted_bed_path)
    
    print(f"{time.time() -t:.3}  Calculating windows", file=sys.stderr)
    # Define a list to store window data
    windows = []
    # Iterate through chromosomes
    for chr_name, chr_seq in (genome.items()):
        # Skip excluded chromosomes, if any
        if exclude and chr_name in exclude:
            continue

        # Calculate the number of windows and window start positions
        
        seq_length = len(chr_seq)
        print(f"{time.time() -t:.3} Calculating number of windows for {chr_name}: {len(chr_seq)}" , file=sys.stderr)
        num_windows = seq_length // window_size  # Number of windows is prescriptive, so the integer division supresses the last shorter window
        window_starts = [i * window_size for i in range(num_windows)]

        # Create windows
        for start in window_starts:
            end = start + window_size
            #end = min(start + window_size, seq_length)
            window_seq = chr_seq[start:end]

            # Calculate GC and AT content
            A_content=window_seq.seq.count("A")
            C_content = window_seq.seq.count("C")
            T_content = window_seq.seq.count("T")
            G_content = window_seq.seq.count("G")
            a_content = window_seq.seq.count("a")
            c_content = window_seq.seq.count("c")
            t_content = window_seq.seq.count("t")
            g_content = window_seq.seq.count("g")
            n_content = window_seq.seq.count("n")
            N_content = window_seq.seq.count("N")
            nN_content = (N_content+n_content) / window_size
            gc_content = (C_content+c_content+G_content+g_content)/window_size
            at_content = (a_content+A_content+T_content+t_content)/window_size

            # Check if the window overlaps with any blacklist regions
            overlaps = ((blacklist_df['chromosome'] == chr_name) &
                        (blacklist_df['start'] < end) &
                        (blacklist_df['end'] > start)).any()

            # Append window information to the list if it does not overlap with the blacklist
            if not overlaps:
                windows.append({
                    'chromosome': chr_name,
                    'start': start+1,
                    'end': end,
                    'GC': gc_content,
                    'AT': at_content,
                    'N': nN_content
                })
            else:
                pass
                #print(f"Skipping window {chr_name}:{start+1}-{end} due to blacklist overlap", file=sys.stderr)
    print("Created windows")

    # Convert the list of dictionaries to a Pandas DataFrame
    windows_df = pd.DataFrame(windows)

    # Return the DataFrame
    return windows_df


def make_windows_vectorized(genome_path : str, blacklisted_bed_path :str, window_size: int, exclude: Union[List[str], None] = None, supress_partial_lastwindow: bool = True) -> pd.DataFrame:
    # Load the genome from a FASTA file

    vals_mapper = np.zeros(256, dtype=np.uint8) + 3
    vals_mapper[ord('a')] = 0
    vals_mapper[ord('A')] = 0
    vals_mapper[ord('t')] = 0
    vals_mapper[ord('T')] = 0
    vals_mapper[ord('g')] = 1
    vals_mapper[ord('G')] = 1
    vals_mapper[ord('c')] = 1
    vals_mapper[ord('C')] = 1
    vals_mapper[ord('n')] = 2
    vals_mapper[ord('N')] = 2
    
    t = time.time()
    print(f"{time.time() -t:.3}  Loading blacklist file", file=sys.stderr)
    blacklist_dict = read_bedfile_chr_dict(blacklisted_bed_path)
    
    total_chr_names = []
    total_starts = []
    total_ends = []
    total_gc_content = []
    total_at_content = []
    total_n_content = []

    for chr_name, chr_seq in tqdm.tqdm(fasta_loader_highmem(genome_path), desc="Chromosomes"):
        # Skip excluded chromosomes, if any
        if exclude and chr_name in exclude:
            continue

        # Calculate the number of windows and window start positions
        print(f"{time.time() -t:.3} Calculating number of windows for {chr_name}" , file=sys.stderr)

        #  loading the string as a numpy array of uint8
        chr_seq = vals_mapper[np.frombuffer(chr_seq.encode('ascii'), dtype=np.uint8)]
        is_at = chr_seq == 0
        at_seq = np.cumsum(is_at) - is_at
        is_gc = chr_seq == 1
        gc_seq = np.cumsum(is_gc) - is_gc

        start_idx = np.arange(0, chr_seq.size, window_size, dtype=np.int32)
        #end_idx = np.arange(window_size-1, chr_seq.size + window_size, window_size, dtype=np.int32)
        end_idx = np.arange(window_size, chr_seq.size + window_size, window_size, dtype=np.int32)
        end_idx[-1] = chr_seq.size - 1

        # Memory complexity is O(blacklist-range-count * window_count)
        blacklist_start = np.array([start for start, _, _ in blacklist_dict[chr_name]], dtype=np.int32)[None, :]
        blacklist_end = np.array([end for _, end, _ in blacklist_dict[chr_name]], dtype=np.int32)[None, :]
        #doesnt_touch = ((end_idx[:, None] < blacklist_start) | (start_idx[:, None] > blacklist_end)).all(axis=1)
        doesnt_touch = ((end_idx[:, None] <= blacklist_start) | (start_idx[:, None] >= blacklist_end)).all(axis=1)

        start_idx = start_idx[doesnt_touch]
        end_idx = end_idx[doesnt_touch]

        window_sizes = np.zeros_like(start_idx) + window_size
        window_sizes[-1] = end_idx[-1] - start_idx[-1]

        if supress_partial_lastwindow and window_sizes[-1] < window_size:
            end_idx = end_idx[:-1]
            start_idx = start_idx[:-1]
            window_sizes = window_sizes[:-1]

        at_content = (at_seq[end_idx] - at_seq[start_idx]) / window_sizes
        gc_content = (gc_seq[end_idx] - gc_seq[start_idx]) / window_sizes
        n_content = 1 - (at_content + gc_content)
        total_starts.append(start_idx)
        total_ends.append(end_idx)
        total_gc_content.append(gc_content)
        total_at_content.append(at_content)
        total_n_content.append(n_content)
        total_chr_names.append(np.array([chr_name] * len(start_idx)))

    window_starts = np.concatenate(total_starts)
    window_ends = np.concatenate(total_ends)
    gc_content = np.concatenate(total_gc_content)
    at_content = np.concatenate(total_at_content)
    n_content = np.concatenate(total_n_content)
    chr_names = np.concatenate(total_chr_names)

    print(f"{time.time() -t:.3}  Creating frame", file=sys.stderr)        
    return pd.DataFrame({
        'chromosome': chr_names,
        'start': window_starts+1,
        'end': window_ends,
        'GC': gc_content,
        'AT': at_content,
        'N': n_content
    })


def main():
    p = {
        "blacklist_file": "./data/hg19.blacklist.v2.bed",
        "genome_file": "./data/hg19.fa",
        "window_size": 100000,
        "exclude": [set(["chrY", "chrX"]), " Which chromosomes to exclude"],
        "vectorized": [False, "Use the vectorized version which is ~40% faster on a window of 100000 and has constant complexity regardless of the window size"],
        "output": ["./data/windows_biopython.csv", "The output csv file path"],
    }

    args, _ = fargv.fargv(p)
    if args.vectorized:
        windows = make_windows_vectorized(args.genome_file, args.blacklist_file, args.window_size, args.exclude)
    else:
        windows = make_windows(args.genome_file, args.blacklist_file, args.window_size, args.exclude)
    windows.to_csv(args.output)


if __name__ == "__main__":
    main()
    #for name, seq in fasta_loader_highmem("./data/hg19.fa"):
    #    #print(name, len(seq), set(seq))_vectorized
    #    print(name, len(seq))
