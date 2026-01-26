import pytest
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
from io import StringIO

from pyEpiAneufinder.makeWindows import make_windows, open_fasta, read_bed_file

def test_read_bed_file(tmp_path):
    """Test reading a BED file into a DataFrame."""
    bed_content = "chr1\t0\t1000\nchr1\t1000\t2000\nchr2\t0\t1000\n"
    bed_file = tmp_path / "test.bed"
    bed_file.write_text(bed_content)

    bed_df = read_bed_file(str(bed_file))

    assert bed_df.shape == (3, 3)
    assert list(bed_df.columns) == ['chromosome', 'start', 'end']
    assert list(bed_df.columns) == ["chromosome","start","end"]
    assert len(bed_df) == 3

def test_open_fasta(tmp_path):
    """Test opening a FASTA file, both plain and gzipped."""
    fasta_content = ">chr1\nACGTACGTACGT\n>chr2\nTGCACTGACTGA\n"
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(fasta_content)

    # Test plain FASTA
    with open_fasta(str(fasta_file)) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        assert len(records) == 2
        assert records[0].id == "chr1"
        assert str(records[0].seq) == "ACGTACGTACGT"

    # Test gzipped FASTA
    gz_fasta_file = tmp_path / "test.fasta.gz"
    with gzip.open(gz_fasta_file, "wt") as f:
        f.write(fasta_content)

    with open_fasta(str(gz_fasta_file)) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        assert len(records) == 2
        assert records[1].id == "chr2"
        assert str(records[1].seq) == "TGCACTGACTGA"


def test_make_windows_simple(tmp_path):
    """Test the make_windows function with a simple genome and no blacklist."""
    # Create small genome
    records = [SeqRecord(Seq("ATGC"*5), id="chr1")]
    genome_file = tmp_path / "genome.fa"
    with open(genome_file, "w") as f:
        SeqIO.write(records, f, "fasta")
    
    # Empty blacklist
    blacklist_file = tmp_path / "blacklist.bed"
    blacklist_file.write_text("")

    df = make_windows(str(genome_file), str(blacklist_file), window_size=4)
    # Genome length 20, window size 4 → 5 windows
    assert len(df) == 5
    assert all(col in df.columns for col in ["chromosome","start","end","GC","AT","N"])

def test_make_windows_with_blacklist(tmp_path):
    """Test the make_windows function with a blacklist."""
    # Create small genome
    records = [SeqRecord(Seq("ATGC"*5), id="chr1")]
    genome_file = tmp_path / "genome.fa"
    with open(genome_file, "w") as f:
        SeqIO.write(records, f, "fasta")
    
    # Blacklist that excludes the second window
    blacklist_content = "chr1\t4\t8\n"
    blacklist_file = tmp_path / "blacklist.bed"
    blacklist_file.write_text(blacklist_content)

    df = make_windows(str(genome_file), str(blacklist_file), window_size=4)
    # Genome length 20, window size 4 → 5 windows, but one excluded → 4 windows
    assert len(df) == 4
    excluded_starts = [4]
    assert all(start not in excluded_starts for start in df['start'])

def test_make_windows_exclude_chromosomes(tmp_path):
    """Test the make_windows function with chromosome exclusion."""
    # Create small genome with two chromosomes
    records = [SeqRecord(Seq("ATGC"*5), id="chr1"),
               SeqRecord(Seq("TGCA"*5), id="chr2")]
    genome_file = tmp_path / "genome.fa"
    with open(genome_file, "w") as f:
        SeqIO.write(records, f, "fasta")
    
    # Empty blacklist
    blacklist_file = tmp_path / "blacklist.bed"
    blacklist_file.write_text("")

    df = make_windows(str(genome_file), str(blacklist_file), window_size=4, exclude=["chr2"])
    # Only chr1 should be present
    assert all(chrom == "chr1" for chrom in df['chromosome'])
    assert len(df) == 5  # chr1 length 20, window size 4 → 5 windows


def test_make_windows_check_GCcontent(tmp_path):
    """Test that GC content is calculated correctly."""
    # Genome sequence: 4 A, 2 T, 2 G, 2 C, 2 N
    # Total length = 12
    seq_str = "AATATAGGNCCN"
    records = [SeqRecord(Seq(seq_str), id="chr1")]
    genome_file = tmp_path / "genome.fa"
    with open(genome_file, "w") as f:
        SeqIO.write(records, f, "fasta")

    # Empty blacklist
    blacklist_file = tmp_path / "blacklist.bed"
    blacklist_file.write_text("")

    # Window size = full sequence
    df = make_windows(str(genome_file), str(blacklist_file), window_size=len(seq_str))

    # Only one window
    assert len(df) == 1
    row = df.iloc[0]

    # GC = (G+C)/window_size = (2+2)/12 = 0.3333
    assert abs(row['GC'] - 0.3333) < 1e-4

    # AT = (A+T)/window_size = (4+2)/12 = 0.5
    assert abs(row['AT'] - 0.5) < 1e-4

    # N = 2 / 12 ≈ 0.1667
    assert abs(row['N'] - 0.1667) < 1e-4

