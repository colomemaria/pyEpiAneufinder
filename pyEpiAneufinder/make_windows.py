import numpy as np
import pandas as pd
from pyranges import PyRanges

def make_windows(genome, blacklist, window_size, exclude=None):
    genome = globals()[genome]  # Assuming genome is a variable holding the genome data.
    seqlengths = genome.seqnames
    windows = PyRanges(seqlengths=seqlengths).tile(window_size, cut_last_tile_in_chrom=True)
    def keep_standard_chromosomes(windows):
        return windows[windows.Chromosome.isin(["chr" + str(i) for i in range(1, 23)])]
    windows = keep_standard_chromosomes(windows)
    if exclude is not None:
        windows = windows[~windows.Chromosome.isin(exclude)]
    windows.df["wSeq"] = windows.Chromosome
    windows.df["wStart"] = windows.Start
    windows.df["wEnd"] = windows.End
    print("Subtracting Blacklist...")
    overlaps = windows.join(blacklist, how="left")
    windowsBL = overlaps[overlaps.Score.isna()].drop("Score")
    print("Adding Nucleotide Information...")
    windowSplit = windowsBL.split("Chromosome")
    def compute_gc_at_N(seq, ranges):
        aFreq = pd.DataFrame(
            np.array([seq[r].count("G") + seq[r].count("C"), seq[r].count("A") + seq[r].count("T")]).T,
            columns=["GC", "AT"],
        )
        ranges.df["GC"] = aFreq["GC"] / (aFreq["GC"] + aFreq["AT"])
        ranges.df["AT"] = aFreq["AT"] / (aFreq["GC"] + aFreq["AT"])
        ranges.df["N"] = 1 - (ranges.df["GC"] + ranges.df["AT"])
        return ranges
    windowNuc = windowSplit.apply(compute_gc_at_N)
    windowNuc = windowNuc.sort("Chromosome", "Start")
    print("Finished making windows successfully")
    return windowNuc