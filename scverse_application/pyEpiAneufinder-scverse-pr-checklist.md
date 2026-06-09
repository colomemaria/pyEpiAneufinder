### Mandatory

Name of the tool: pyEpiAneufinder

Short description:
pyEpiAneufinder identifies copy number variations (CNVs) from single-cell
ATAC-seq data and provides utilities for CNV visualization, subclone
annotation, and karyotype-level summary metrics.

How does the package use scverse data structures (please describe in a few sentences):
pyEpiAneufinder uses `AnnData` internally to represent the binned count matrix,
stores this intermediate as `count_matrix.h5ad`, and reuses it in the workflow
resume path and in `plot_single_cell_profile`. This provides interoperability
with the scverse ecosystem through an `AnnData`-based data object and `.h5ad`
output.

- [x] The code is publicly available under an OSI-approved license
- [x] The package provides versioned releases
- [x] The package can be installed from a standard registry (e.g. PyPI, conda-forge, bioconda)
- [x] Automated tests cover essential functions of the package and a reasonable range of inputs and conditions
- [x] Continuous integration (CI) automatically executes these tests on each push or pull request
- [x] The package provides API documentation via a website or README
- [x] The package uses scverse datastructures where appropriate (i.e. AnnData, MuData or SpatialData and their modality-specific extensions)
- [x] I am an author or maintainer of the tool and agree on listing the package on the scverse website

### Recommended

- [ ] Please announce this package on scverse communication channels (zulip, discourse, twitter)

- [ ] Please tag the author(s) these announcements. Handles (e.g. `@scverse_team`) to include are:

  Zulip:
  Discourse:
  Mastodon:
  Bluesky:
  Twitter:

- [x] The package provides tutorials (or "vignettes") that help getting users started quickly

- [ ] The package uses the scverse cookiecutter template.
