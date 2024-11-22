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

    # ----------------------------------------------------------------------- 
    # Estimating break points
    # ----------------------------------------------------------------------- 

    #Assumption: count matrix as anndata object (might need to be changed later)
    print("Calculating distance AD")

    unique_chroms=counts.var["chr"].unique()

    cluster_ad = {}
    for i in range(counts.shape[0]):
        cell_name = counts.obs.cellID[i]
        cluster_ad[cell_name]=pd.DataFrame()
        for chrom in unique_chroms:
            #Identify the breakpoints
            bp_chrom=getbp(counts.X[i,counts.var["chr"]==chrom].toarray().flatten(),
                           k=k,minsize=minsize,minsizeCNV=minsizeCNV)
            bp_chrom["chr"]=chrom
        
            #Merge the pandas data frames across chromosomes to one per cell
            cluster_ad[cell_name] = pd.concat([cluster_ad[cell_name],bp_chrom],axis=0,ignore_index=True)
            
    print("Successfully identified breakpoints")

    # -----------------------------------------------------------------------    
    # Pruning break points and annotating CNV status of each segment
    # ----------------------------------------------------------------------- 

    #Prune irrelevant breakpoints
    clusters_pruned = {}
    for cell, bp_frame in cluster_ad.items():
        clusters_pruned[cell] = threshold_dist_values(bp_frame)

    print("Successfully discarded irrelevant breakpoints")

    # ----------------------------------------------------------------------- 
    # Plot the result as a karyogram
    # ----------------------------------------------------------------------- 

    if(plotKaryo):
        plot_karyo_gainloss()