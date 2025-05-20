import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import time 
import os

from .makeWindows import make_windows
from .render_fragments import process_fragments, get_loess_smoothed
from .get_breakpoints import getbp
from .assign_somy import threshold_dist_values, assign_gainloss
from .plotting import karyo_gainloss

def epiAneufinder(input, outdir, genome_file,
                  blacklist, windowSize,
                  test='AD', reuse_existing=False, exclude=None,
                  uq=0.9, lq=0.1, title_karyo=None, minFrags = 20000,
                  threshold_cells_nbins=0.05,selected_cells=None,
                  threshold_blacklist_bins=0.85,
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

    #Create the output dir if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    # ----------------------------------------------------------------------- 
    # Create windows from genome file (with GC content per window)
    # ----------------------------------------------------------------------- 

    print("Binning the genome")

    start = time.perf_counter()

    windows_file_name = outdir+"/binned_genome.csv"

    windows = make_windows(genome_file, blacklist, windowSize, exclude)
    windows.to_csv(windows_file_name)

    end = time.perf_counter()
    execution_time = (end - start)/60
    print(f"Successfully binned the genome. Execution time: {execution_time:.2f} mins")

    # ----------------------------------------------------------------------- 
    # Read the fragment file and generate a count matrix from it
    # ----------------------------------------------------------------------- 

    print("Reading fragment file")

    start = time.perf_counter()

    counts = process_fragments(windows_file_name,input,windowSize, minFrags)

    end = time.perf_counter()
    execution_time = (end - start)/60
    print(f"Successfully read fragment file. Execution time: {execution_time:.2f} mins")

    # -----------------------------------------------------------------------
    # Filtering cells
    # -----------------------------------------------------------------------

    #Exclude cells that have no signal in most bins
    nonzero_cell = counts.X.getnnz(axis=1)
    filter_cells = nonzero_cell > threshold_cells_nbins * counts.X.shape[1]
    counts = counts[filter_cells,:].copy()
    print(f"Filtering cells without enough coverage, {counts.X.shape[0]} cells remain.")

    #Exclude bins that have no signal in most cells
    nonzero_bins = counts.X.getnnz(axis=0)
    filter_bins = nonzero_bins >= (1- threshold_blacklist_bins) * counts.X.shape[0]
    counts = counts[:,filter_bins].copy()
    print(f"Filtering windows without enough coverage, {counts.X.shape[1]} windows remain.")

    # ----------------------------------------------------------------------- 
    # GC correction
    # ----------------------------------------------------------------------- 

    print("GC correction")

    start = time.perf_counter()

    #Perform GC correction per cell
    all_loess_rows = []
    for i in range(counts.X.shape[0]):
        counts_per_window=counts.X[i,:].toarray().flatten()
        loess_res = get_loess_smoothed(counts_per_window, counts.var.GC.to_numpy())
        correction = counts_per_window.mean()/loess_res #(loess_res + .000000000001)
        loess_norm_row = counts_per_window * correction
        #Round to integer again (speeds runtime significantly!)
        all_loess_rows.append(np.rint(loess_norm_row).astype(int))


    #Keep the raw data
    counts.layers["raw"]=counts.X

    #Save the GC normalized matrix in X
    counts.X = csr_matrix(np.vstack(all_loess_rows))

    end = time.perf_counter()
    execution_time = (end - start)/60
    print(f"Successfully performed GC correction. Execution time: {execution_time:.2f} mins")

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
            bp_chrom = breakpoints_pruned.breakpoint[(breakpoints_pruned.seq==chrom) & 
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