# pyEpiAneufinder: Identifying copy number alterations from single-cell ATAC-seq data

[![PyPI](https://img.shields.io/pypi/v/pyEpiAneufinder.svg?color=blue&label=PyPi)](https://pypi.org/project/pyEpiAneufinder/)
[![GitHub release](https://img.shields.io/github/v/release/colomemaria/pyEpiAneufinder?color=blue&label=release)](https://github.com/colomemaria/pyEpiAneufinder/releases)
[![Tests](https://github.com/colomemaria/pyEpiAneufinder/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/colomemaria/pyEpiAneufinder/actions/workflows/tests.yml)
[![Docs CI](https://github.com/colomemaria/pyEpiAneufinder/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/colomemaria/pyEpiAneufinder/actions/workflows/docs.yml)
[![Read the Docs](https://img.shields.io/badge/Read%20the%20Docs-online-brightgreen)](https://pyepianeufinder.readthedocs.io/)

This package is the Python re-implementation of our R package
`epiAneufinder`, developed for identifying copy number variations (CNVs) from
single-cell ATAC-seq data.

**Important remark:** The Python package is still in beta testing. Please
report issues and improvement suggestions through GitHub Issues.

Single-cell open chromatin profiling through scATAC-seq provides fragment count
information that can be used to infer copy number variation. `pyEpiAneufinder`
uses this information to extract genome-wide CNV profiles for individual cells,
adding a layer of genomic variation analysis without requiring additional
experiments.

The original `epiAneufinder` publication is:

Ramakrishnan, A., Symeonidi, A., Hanel, P. et al. epiAneufinder identifies copy
number alterations from single-cell ATAC-seq data. *Nature Communications* 14,
5846 (2023). https://doi.org/10.1038/s41467-023-41076-1

The R version, including additional background information, is available at:
https://github.com/colomemaria/epiAneufinder

### Installation

```bash
pip install pyEpiAneufinder
```

For development, including tests and docs:

```bash
pip install -e ".[test,docs]"
```

The full workflow guide, figures, API usage, and example analyses are available
in the Read the Docs documentation:
https://pyepianeufinder.readthedocs.io/

### Cite

If you use `pyEpiAneufinder`, please cite both the `epiAneufinder` publication.

Ramakrishnan, A., Symeonidi, A., Hanel, P. et al. epiAneufinder identifies copy
number alterations from single-cell ATAC-seq data. *Nature Communications* 14,
5846 (2023). https://doi.org/10.1038/s41467-023-41076-1

### Authors

Katharina Schmid (katharina.schmid@bmc.med.lmu.de)

Ida Bueschel (Ida.Bueschel@helmholtz-munich.de)

Aikaterini Symeonidi (asymeonidi@bmc.med.lmu.de and ksymeonidh@gmail.com)

Muhammet A. Celik

Angelos Nikolaou

Maria Colome-Tatche (maria.colome@bmc.med.lmu.de)

### Version history

* 0.3.7
    * Test release of workflows after merging PR
* 0.3.6
    * Test release for GitHub Release and PyPI publishing workflows
* 0.1
    * Initial Release (based on epiAneufinder v1.1.3)
