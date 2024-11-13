"""Main function of epiAneufinder

Runs all the necessary steps based on an input file, namely creating a binned count matrix,
filtering, GC correction, estimation of breakpoints, pruning, annotation of segments and plotting.
"""

def epiAneufinder(input, outdir, blacklist, windowSize, genome="BSgenome.Hsapiens.UCSC.hg38",
                    test='AD', reuse_existing=False, exclude=None,
                    uq=0.9, lq=0.1, title_karyo=None, minFrags = 20000, mapqFilter=10,
                    threshold_cells_nbins=0.05,selected_cells=None,
                    gc_correction="loess",threshold_blacklist_bins=0.85,
                    ncores=4, minsize=1, k=4, 
                    minsizeCNV=0,plotKaryo=True):

    # Create the count matrix

    # Filtering cells

    # GC correction

    # Estimating break points

    #Pruning break points and annotating CNV status of each segment

    # Plot the result as a karyogram
    if(plotKaryo):
        plot_karyo_gainloss()