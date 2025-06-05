# pyEpiAneufinder: Identifying copy number alterations from single-cell ATAC-seq data

This package is the python version of our R package epiAneufinder (based on version 1.1.1). The python package is still in beta-testing and contains still reduced parameter options compared to the R version. Please report any issues and improvement suggestions via github.

epiAneufinder is an algorithm used for calling Copy Number Variations (CNVs) from single-cell ATAC (scATAC) data. Single-cell open chromatin profiling via the single-cell Assay for Transposase-Accessible Chromatin using sequencing (scATAC-seq) assay has become a mainstream measurement of open chromatin in single-cells. epiAneufinder exploits the read count information from scATAC-seq data to extract genome-wide copy number variations (CNVs) for each individual cell. epiAneufinder allows the addition of single-cell CNV information to scATAC-seq data, without the need of additional experiments, unlocking a layer of genomic variation which is otherwise unexplored.

Ramakrishnan, A., Symeonidi, A., Hanel, P. et al. epiAneufinder identifies copy number alterations from single-cell ATAC-seq data. Nat Commun 14, 5846 (2023). https://doi.org/10.1038/s41467-023-41076-1

The R version (including more information) can be found here: https://github.com/colomemaria/epiAneufinder

### Installation

Potentially setup a new conda environment first (recommended). Then:

```
pip install git+https://github.com/colomemaria/pyEpiAneufinder
```

### Executing the program

The whole program can be run by calling the main function, using a fragment file as input, defined in the parameter `fragment_file`. It
saves all output files in `outdir`. 

The `genome` needs to be  given as a fasta file. For example, the human genome hg38 can be downloaded from here:
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

Additionally, a `blacklist` file for regions of low mappability is required, the `windowSize` of the algorithms and optional a list of chromosomes to `exclude`.
Cells with too little fragments are removed based on the parameter `minFrags`.

```
import pyEpiAneufinder as pea
pea.epiAneufinder(fragment_file="sample_data/sample.tsv", 
                  outdir="results_sample_data", 
                  genome_file="hg38.fa.gz", 
                  blacklist="sample_data/hg38-blacklist.v2.bed",
                  windowSize=100000, 
                  exclude = ["chrX","chrY"],
                  minFrags=20000)
```

### Authors of the python re-implementation

Katharina Schmid (katharina.schmid@bmc.med.lmu.de)

Aikaterini Symeonidi (asymeonidi@bmc.med.lmu.de and ksymeonidh@gmail.com)

Angelos Nikolaou (anguelos.nicolaou@gmail.com)

Maria Colomé-Tatché (maria.colome@helmholtz-muenchen.de)

### Version history

* 0.1
    * Initial Release (based on epiAneufinder v1.1.1)
