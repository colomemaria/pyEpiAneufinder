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
    breakpoints_pruned = {}
    for cell, bp_frame in cluster_ad.items():
        
        breakpoints_pruned[cell] = threshold_dist_values(bp_frame)

    print("Successfully discarded irrelevant breakpoints")

    #Number of bins per chromosome
    num_bins_chrom = counts.var["chr"].value_counts(sort=False)

    #Convert breakpoints into segment annotations per cell
    clusters_pruned={}
    for cell, bp_frame in breakpoints_pruned.items():
        
        print(cell)
        
        counter=1
        cluster_list=[]
        
        for chrom in unique_chroms:
            
            #Extract all breakpoints from this chromosome
            bp_chrom = bp_frame.breakpoint[bp_frame.chr==chrom]
            
            #If no breakpoints exist for this chromsome, save all windows as one segment
            if bp_chrom.empty:
                cluster_list += [counter] * num_bins_chrom[chrom]
            else:
                
                #Otherwise calculate the length of each segment (in the right order)
                bp_chrom = sorted([0,num_bins_chrom[chrom]]+bp_chrom.tolist())
                segment_size = np.diff(bp_chrom)
                
                #Add for each segment the indices
                cluster_list += np.repeat(range(counter,len(segment_size)+counter),segment_size).tolist()
            
            #Decide which index the next segment should get
            counter = max(cluster_list)+1
    
        #Save all indices as a new dictonary entry
        clusters_pruned[cell]=cluster_list

    #Assign the somies for each cell
    somies_ad={}
    for cell, cluster_cell in clusters_pruned.items():
        somies_ad[cell]=assign_gainloss(counts_filtered.X[counts_filtered.obs.cellID == cell,].toarray().flatten(),
                                        cluster_cell,uq,lq)
        
    # ----------------------------------------------------------------------- 
    # Plot the result as a karyogram
    # ----------------------------------------------------------------------- 

    if(plotKaryo):
        plot_karyo_gainloss()