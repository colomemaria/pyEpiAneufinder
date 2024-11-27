import pandas as pd
import numpy as np
import time 

from .get_breakpoints import getbp
from .assign_somy import threshold_dist_values, assign_gainloss
from .plotting import karyo_gainloss

def epiAneufinder(input, outdir, blacklist, windowSize, genome="BSgenome.Hsapiens.UCSC.hg38",
                    test='AD', reuse_existing=False, exclude=None,
                    uq=0.9, lq=0.1, title_karyo=None, minFrags = 20000, mapqFilter=10,
                    threshold_cells_nbins=0.05,selected_cells=None,
                    gc_correction="loess",threshold_blacklist_bins=0.85,
                    ncores=4, minsize=1, k=4, 
                    minsizeCNV=0,plotKaryo=True):

    """
    Main function of epiAneufinder

    Runs all the necessary steps based on an input file, namely creating a binned count matrix,
    filtering, GC correction, estimation of breakpoints, pruning, annotation of segments and plotting.

    Parameters
    ----------
    input: Folder with bam files, a fragments.tsv/bed file or a folder with a count matrix (required files: matrix.mtx(.gz), barcodes.tsv(.gz) and peaks.bed(.gz))
    outdir: Path to output directory
    blacklist: Bed file with blacklisted regions
    windowSize: Size of the window (Reccomended for sparse data - 1e6)
    genome: String containing name of BS.genome object. Necessary for GC correction. Default: "BSgenome.Hsapiens.UCSC.hg38"
    test: Currently only "AD" (=Anderson-Darling) is implemented.
    reuse_existing: Logical. False removes all the files in the outdir and recomputes everything.
    exclude: String of chromosomes to exclude. Example: c('chrX','chrY','chrM')
    uq: Upper quantile. Default: 0.1
    lq: Lower quantile. Default: 0.9
    title_karyo: String. Title of the output karyogram
    minFrags: Integer. Minimum number of reads for a cell to pass. Only required for fragments.tsv file. Default: 20000
    mapqFilter: Filter bam files after a certain mapq value
    threshold_cells_nbins: Keep only cells that have more than a certain percentage of non-zero bins
    selected_cells: Additional option for filtering the input, either NULL or a file with barcodes of cells to keep (one barcode per line, no header)
    gc_correction: Type of GC correction, currently implemented options are "loess", "bulk_loess" and "quadratic".
    threshold_blacklist_bins: Blacklist a bin if more than the given ratio of cells have zero reads in the bin. Default: 0.85
    ncores: Number of cores for parallelization. Default: 4
    minsize: Integer. Resolution at the level of ins. Default: 1. Setting it to higher numbers runs the algorithm faster at the cost of resolution
    k: Integer. Find 2^k segments per chromosome
    minsizeCNV: Integer. Number of consecutive bins to constitute a possible CNV
    plotKaryo: Boolean variable. Whether the final karyogram is plotted at the end

    Output
    ------
    csv file with predictions, png file with karyogram (if plotKaryo=True) as well as intermediate results
    
    """

    # Create the count matrix

    # Filtering cells

    # GC correction

    # ----------------------------------------------------------------------- 
    # Debugging code to check the second part of the function
    # ----------------------------------------------------------------------- 

    import scanpy as sc
    from scipy.sparse import csr_matrix

    #Read GC corrected counts and annotations (from R version of epiAneufinder)
    annot=pd.read_csv(input+"/sample_pea_annot.csv",sep=" ")

    gc_counts=pd.read_csv("/Users/kschmid/Desktop/sample_pea_count.csv",sep=" ")
    gc_counts = gc_counts.T
    gc_counts.columns = ["win"+str(v) for v in list(gc_counts.columns.values)]

    counts = sc.AnnData(gc_counts)

    #Add window annotation
    counts.var["seq"]=list(annot["seqnames"])
    counts.var["start"]=list(annot["start"])
    counts.var["end"]=list(annot["end"])
    counts.obs["cellID"] = counts.obs.index.to_list()

    #Convert matrix into a sparse matrix
    counts.X = csr_matrix(counts.X)

    # ----------------------------------------------------------------------- 
    # Estimating break points
    # ----------------------------------------------------------------------- 

    #Assumption: count matrix as anndata object (might need to be changed later)
    print("Calculating distance AD")

    start = time.perf_counter()

    unique_chroms=counts.var["seq"].unique()

    cluster_ad = pd.DataFrame()
    for i in range(counts.shape[0]):
        cell_name = counts.obs.cellID[i]
        for chrom in unique_chroms:
            #Identify the breakpoints
            bp_chrom=getbp(counts.X[i,counts.var["seq"]==chrom].toarray().flatten(),
                           k=k,minsize=minsize,minsizeCNV=minsizeCNV)
            bp_chrom["cell"]= cell_name
            bp_chrom["seq"]=chrom
        
            #Merge the pandas data frames across chromosomes to one per cell
            cluster_ad = pd.concat([cluster_ad,bp_chrom],axis=0,ignore_index=True)

    #Save the found breakpoints
    cluster_ad.to_csv(outdir+"/breakpoints_unfiltered.csv")

    end = time.perf_counter()
    execution_time = (end - start)/60
    print(f"Successfully identified breakpoints. Execution time: {execution_time:.2f} mins")

    # -----------------------------------------------------------------------    
    # Pruning break points and annotating CNV status of each segment
    # ----------------------------------------------------------------------- 

    print("Prunning breakpoints")

    start = time.perf_counter()

    #Prune irrelevant breakpoints
    breakpoints_pruned = pd.DataFrame()
    for cell in cluster_ad["cell"].unique():
        clusters_cell=cluster_ad[cluster_ad.cell==cell].copy()
        bp_cell = threshold_dist_values(clusters_cell)

        #Merge the pandas data frames across chromosomes to one per cell
        breakpoints_pruned = pd.concat([breakpoints_pruned,bp_cell],axis=0,ignore_index=True)

    #Save the pruned breakpoints
    breakpoints_pruned.to_csv(outdir+"/breakpoints_pruned.csv")

    end = time.perf_counter()
    execution_time = (end - start)/60
    print(f"Successfully discarded irrelevant breakpoints. Execution time: {execution_time:.2f} mins")

    print("Assign somies")

    start = time.perf_counter()

    #Number of bins per chromosome
    num_bins_chrom = counts.var["seq"].value_counts(sort=False)

    #Convert breakpoints into segment annotations per cell
    clusters_pruned={}
    for cell in breakpoints_pruned["cell"].unique():
        
        counter=1
        cluster_list=[]
        
        for chrom in unique_chroms:

            #Extract all breakpoints from this chromosome
            bp_chrom = breakpoints_pruned.breakpoint[(breakpoints_pruned.chr==chrom) & 
                                                     (breakpoints_pruned.cell==cell) ]
            
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
    somies_ad=counts.var[["seq","start","end"]]
    for cell, cluster_cell in clusters_pruned.items():
        somies_ad[cell]=list(assign_gainloss(counts.X[counts.obs.cellID == cell,].toarray().flatten(),cluster_cell))

    end = time.perf_counter()
    execution_time = (end - start)/60
    print(f"Successfully identified somies. Execution time: {execution_time:.2f} mins")


    #Save the results as a tsv file
    somies_ad.to_csv(outdir+"/result_table.csv")

    print("""A .tsv file with the results has been written to disk. 
          It contains the copy number states for each cell per bin. 
          0 denotes 'Loss', 1 denotes 'Normal', 2 denotes 'Gain'.""")

    # ----------------------------------------------------------------------- 
    # Plot the result as a karyogram
    # ----------------------------------------------------------------------- 

    if(plotKaryo):

        start = time.perf_counter()

        karyo_gainloss(somies_ad,outdir,title_karyo)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully plotted karyogram. Execution time: {execution_time:.2f} mins")