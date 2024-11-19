from Bio import SeqIO
from Bio.SeqUtils import nt_search
import pandas as pd

def read_bed_file(bed_file):
    # Read the BED file into a DataFrame
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chromosome', 'start', 'end'],index_col=False)
    return bed_df


def make_windows(genome_file, bed_file, window_size, exclude=None):
    # Load the genome from a FASTA file
    print("Loading genome file")
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
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

        # Calculate the number of windows and window start positions
        seq_length = len(chr_seq)
        num_windows = seq_length // window_size
        window_starts = [i * window_size for i in range(num_windows)]
        print("Calculating number of windows")

        # Create windows
        for start in window_starts:
            end = start + window_size
            window_seq = chr_seq[start:end]

            # Calculate GC and AT content
            #c_content=nt_search(window_seq,"C")
            #g_content = nt_search(window_seq, "G")
            #a_content = nt_search(window_seq, "A")
            #t_content = nt_search(window_seq, "T")
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

    # Convert the list of dictionaries to a Pandas DataFrame
    windows_df = pd.DataFrame(windows)

    # Return the DataFrame
    return windows_df

# Example usage:
# genome_file = "path_to_hg38_genome.fasta"
# bed_file = "path_to_blacklist.bed"
# window_size = 1000
# exclude = ["chrX", "chrY"]  # Chromosomes to exclude
# windows = make_windows(genome_file, bed_file, window_size, exclude)
if __name__ =="__main__":
    blacklist_file="/home/katia/Helmholz/epiAneufinder/revisions/hg19.blacklist.v2.bed"
    genome_file="/home/katia/Helmholz/epiAneufinderPython/hg19/hg19.fa"
    #genome_file = "test.fasta"
    window_size=100000
    exclude=["chrY","chrX"]
    windows=make_windows(genome_file, blacklist_file, window_size, exclude)
    windows.to_csv("hg19_windows.csv")
