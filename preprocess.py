#!/usr/bin/env python3

import sys
from typing import Dict, Generator, List, Tuple
import fargv
import gzip
import tqdm
from collections import defaultdict


def load_lines(filename: str, chr2int: Dict[str, int]) -> Generator[Tuple[str, int, int, int, int], None, None]:
    """Generator to reduce memory usage by loading lines from a file one at a time.

    Args:
        filename (str): A tsv or tsv.gz file with columns: chromosome, start, end, cell name, read support.
        chr2int (Dict[str, int]): A dictionary mapping chromosome names to integers

    Yields:
        Generator[Tuple[str, int, int, int, int], None, None]: A generator that yields tuples of cell name,
            chromosome, start, end, read support
    """
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    else:
        f = open(filename, "r")

    for line in f.readlines():
        if line.startswith("#"):
            continue
        line = line.split("\t")
        if len(line) != 5:
            print(f"Error: {line}", file=sys.stderr)
            continue
        yield line[3], chr2int[line[0]], int(line[1]), int(line[2]), int(line[4])


def render_cell(cell_name: str, fragments: List[Tuple[int, int, int, int]], int2chr: Dict[int, str]) -> str:
    """Writes all fragments for a cell to a tsv string

    Args:
        cell_name (str): The cell barcode
        fragments (List[Tuple[int, int, int, int]]): A list of tuples with chromosome, start, end, and fragment count
        int2chr (Dict[int, str]): A dictionary mapping integers to chromosome names

    Returns:
        str: A string with tsv formatted lines of chromosome, start, end, cell barcode,  fragment_count.
            The last line will not have a newline so it should be added after this string.
    """
    fragments = sorted(fragments)  # Sort fragments by chromosome then start position then end position
    res_lines = []
    for fragment in fragments:
        res_lines.append("\t".join([int2chr[fragment[0]], str(fragment[1]), str(fragment[2]), cell_name, str(fragment[3])]))
    return "\n".join(res_lines)


def main():
    """Main function to preprocess fragments from a tsv file and generate a new one."""
    p = {
        "filename": "./data/GSM3722064_SU008_Tumor_Pre_fragments.tsv.gz",
        "output": "output.tsv.gz",
        "min_fragments": 10000,
        "chr_exclude": set([]),
        "genome": "hg19",
    }

    args, _ = fargv.fargv(p)
    if args.genome in ["hg19", "hg38"]:
        chr2int = {f"chr{n}": n for n in range(0, 23)}
        chr2int.update({"chrM": 24, "chrX": 25, "chrY": 26})
        int2chr = {v: k for k, v in chr2int.items()}
    elif args.genome == "mm10":
        chr2int = {f"chr{n}": n for n in range(0, 20)}
        chr2int.update({"chrM": 20, "chrX": 21, "chrY": 22})
        int2chr = {v: k for k, v in chr2int.items()}
    else:
        raise ValueError(f"Unknown genome {args.genome}")

    chr_exclude = [chr2int[chr] for chr in args.chr_exclude]
    chr_include = [chr2int[c] for c in (set(chr2int) - set(chr_exclude))]
    print(f"Excluding Chromosomes {sorted(args.chr_exclude)}\nIncluding chromosomes {[int2chr[c] for c in chr_include]}", file=sys.stderr)

    # Here begins the memory bottle neck
    by_cell = defaultdict(lambda: [])
    excluded = 0
    for line in tqdm.tqdm(load_lines(filename=args.filename, chr2int=chr2int), "Loading input fragments"):
        if line[1] in chr_include:
            by_cell[line[0]].append(line[1:])
        else:
            excluded += 1
    print(f"Found {len(by_cell)} cells, excluded {excluded} fragments from excluded chromosomes", file=sys.stderr)

    by_cell = {k: v for k, v in tqdm.tqdm(by_cell.items(), "Filtering by fragment count") if len(v) >= args.min_fragments}
    # This is the end of the memory bottle neck
    print(f"Found {len(by_cell)} cells with at least {args.min_fragments} fragments", file=sys.stderr)
    sys.stderr.flush()

    print(f"Writing output to {args.output}", file=sys.stderr)
    if args.output == "stdout":
        for cell_name, fragments in by_cell.items():
            print(render_cell(cell_name, fragments, int2chr))
    elif args.output.endswith(".tsv.gz"):
        with gzip.open(args.output, "wt") as f:
            for cell_name, fragments in by_cell.items():
                f.write(render_cell(cell_name, fragments, int2chr)+"\n")
    elif args.output.endswith(".tsv"):
        with open(args.output, "w", encoding="utf-8") as f:
            for cell_name, fragments in by_cell.items():
                f.write(render_cell(cell_name, fragments, int2chr)+"\n")
    else:
        raise ValueError(f"Unknown output format {args.output}")

if __name__ == "__main__":
    main()
