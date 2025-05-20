from Bio import SeqIO
from Bio.SeqUtils import nt_search
import pandas as pd
import gzip
import natsort

def read_bed_file(bed_file): # Read the BED file into a DataFrame (only the first three are required)
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, 
                         usecols=[0, 1, 2], 
                         names=['chromosome', 'start', 'end'],index_col=False)
    return bed_df

def open_fasta(filename):
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt")
    else:
        return open(filename, "r")

def make_windows(genome_file, bed_file, window_size, exclude=None):
    # Load the genome from a FASTA file
    print("Loading genome file")

    with open_fasta(genome_file) as handle:
        genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

    print("Genome file loaded")

    # Read the blacklist BED file
    print("Loading blacklist file")
    blacklist_df = read_bed_file(bed_file)
    print("Blacklist file loaded")

    # Define a list to store window data
    windows = []

    # Iterate through chromosomes
    for chr_name, chr_seq in genome.items():
        # Skip excluded chromosomes, if any
        if exclude and chr_name in exclude:
            continue

        #Remove non-standard chromosomes
        if "_" in chr_name:
            continue

        # Calculate the number of windows and window start positions
        #print("Calculating number of windows for "+ chr_name)
        seq_length = len(chr_seq)
        num_windows = seq_length // window_size
        window_starts = [i * window_size for i in range(num_windows)]
        

        # Create windows
        for start in window_starts:
            end = start + window_size
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
            total_N=(N_content+n_content)//window_size
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
                    'N': total_N
                })
    print("Created windows")

    # Convert the list of dictionaries to a Pandas DataFrame
    windows_df = pd.DataFrame(windows)

    #Sort it using the natual sort so that chr2 before chr11
    windows_df.sort_values(by=["chromosome","start"],key=natsort.natsort_keygen(),inplace=True)

    # Return the DataFrame
    return windows_df
