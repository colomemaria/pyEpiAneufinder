import pandas as pd
import numpy as np
import anndata as ad
import scipy as sp
from scipy.sparse import csr_matrix
import time 
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed #heavy CPU
from concurrent.futures import ThreadPoolExecutor, as_completed #heavy IO
from tqdm import tqdm

from .makeWindows import make_windows
from .render_fragments import process_fragments, get_loess_smoothed, process_count_matrix
from .get_breakpoints import fast_getbp, recursive_getbp_df
from .assign_somy import assign_gainloss, assign_gainloss_new
from .plotting import karyo_gainloss

def _process_recursive_bp_worker(args):
    cell, chrom, data_slice_chr, data_slice_cell, k, n_permutations, alpha = args
    bp = recursive_getbp_df(data_slice_chr, data_slice_cell, k=k, n_permutations=n_permutations, alpha=alpha)
    # cell, chrom, data_slice, k, n_permutations, alpha = args
    # bp = recursive_getbp_df(data_slice, k=k, n_permutations=n_permutations, alpha=alpha)
    bp["cell"], bp["seq"] = cell, chrom
    return bp

def epiAneufinder(fragment_file, mode, outdir, genome_file,
                  blacklist, windowSize,
                  exclude=None, sort_fragment=True, GC=True,
                  title_karyo=None, minFrags = 20000,
                  threshold_cells_nbins=0.05,
                  threshold_blacklist_bins=0.85,
                  ncores=1, k=4, 
                  n_permutations=500, alpha=0.05,
                  plotKaryo=True, 
                  resume=False, cellRangerInput=False,
                  remove_barcodes=None,
                  selected_cells=None):

    """
    Main function of epiAneufinder

    Runs all the necessary steps based on an input file, namely creating a binned count matrix,
    filtering, GC correction, estimation of breakpoints, pruning, annotation of segments and plotting.

    Parameters
    ----------
    fragment_file: Folder with bam files, a fragments.tsv/bed file or a folder with a count matrix (required files: matrix.mtx(.gz), barcodes.tsv(.gz) and peaks.bed(.gz))
    mode: String. Choose between "Watson" (conservative) and "Holmes" (explorative)
    outdir: Path to output directory
    blacklist: Bed file with blacklisted regions
    windowSize: Size of the window (Reccomended for sparse data - 1e6)
    genome: String containing name of BS.genome object. Necessary for GC correction. Default: "BSgenome.Hsapiens.UCSC.hg38"
    exclude: String of chromosomes to exclude. Example: c('chrX','chrY','chrM')
    title_karyo: String. Title of the output karyogram
    minFrags: Integer. Minimum number of reads for a cell to pass. Only required for fragments.tsv file. Default: 20000
    GC: Boolean variable. Whether to perform GC correction
    sort_fragment: Boolean variable. Whether to sort the fragment file by cell (can take a while). Default: True
    threshold_cells_nbins: Keep only cells that have more than a certain percentage of non-zero bins
    selected_cells: Additional option for filtering the input, either NULL or a file with barcodes of cells to keep (one barcode per line, no header)
    threshold_blacklist_bins: Blacklist a bin if more than the given ratio of cells have zero reads in the bin. Default: 0.85
    ncores: Number of cores for parallelization. Default: 4
    k: Integer. Find 2^k segments per chromosome
    plotKaryo: Boolean variable. Whether the final karyogram is plotted at the end
    resume : Boolean variable. Whether to resume the analysis if the intermmediate/output files already exist
    cellRangerInput: Boolean variable. Whether the input is a cell ranger output (in this case, the fragment file is not needed and the count matrix is used directly)
    remove_barcodes: Path to TSV file containing barcodes to exclude (one per line)

    Output
    ------
    csv file with predictions, png file with karyogram (if plotKaryo=True) as well as intermediate results
    
    """

    #Create the output dir if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    #Load the barcodes to exclude if provided
    barcodes_to_remove = None
    if remove_barcodes is not None:
        barcodes_to_remove = pd.read_csv(remove_barcodes, header=None)[0].tolist()
        print(f"Loaded {len(barcodes_to_remove)} barcodes to exclude.")
    
    #Load the barcodes to include if provided
    if selected_cells is not None:
        selected_cells = pd.read_csv(selected_cells, header=None)[0].tolist()
        print(f"Loaded {len(selected_cells)} barcodes to include.")


    # ----------------------------------------------------------------------- 
    # Create windows from genome file (with GC content per window)
    # ----------------------------------------------------------------------- 

    print("Binning the genome")

    start = time.perf_counter()

    windows_file_name = outdir+"/binned_genome.csv"
    
    if resume and os.path.exists(windows_file_name): #resume if the windows already exist
        print(f"Resuming: {windows_file_name} already exists. Skipping calculation.")
    else:
        windows = make_windows(genome_file, blacklist, windowSize, exclude)
        windows.to_csv(windows_file_name)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully binned the genome. Execution time: {execution_time:.2f} mins")

    matrix_file = outdir+"/count_matrix.h5ad"
    if resume and os.path.exists(matrix_file):
        print(f"Resuming: {matrix_file} already exists. Skipping calculation.")
        counts = ad.read_h5ad(matrix_file)
    else:
        if cellRangerInput:
            print("Using cell ranger input. No fragment file needed.")
            counts = process_count_matrix(windows_file_name,minFrags,fragment_file,remove_barcodes=barcodes_to_remove, selected_cells=selected_cells)  
        else:      
            # ----------------------------------------------------------------------- 
            # Sort the fragment file by cell (run over shell)
            # ----------------------------------------------------------------------- 

            if sort_fragment:
                print("Sort the fragment file")
                start = time.perf_counter()
                output_file = outdir + "/fragment_file.sortedbycell.tsv.gz"
                cmd = f"zgrep -v '^#' {fragment_file} | sort -k4,4 -k1,1 -k2,2n --parallel={ncores} | gzip > {output_file}"
                subprocess.run(cmd, shell=True, check=True)
                end = time.perf_counter()
                execution_time = (end - start)/60
                print(f"Successfully sort fragment file. Execution time: {execution_time:.2f} mins")
            else:
                print("Taking fragment file correctly sorted by the user after barcode and position")
                output_file = fragment_file


            # ----------------------------------------------------------------------- 
            # Read the fragment file and generate a count matrix from it
            # ----------------------------------------------------------------------- 

            print("Reading the sorted fragment file")

            start = time.perf_counter()

            counts = process_fragments(windows_file_name,output_file,windowSize, minFrags,remove_barcodes=barcodes_to_remove, selected_cells=selected_cells)

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
        if not GC:
            print("Skipping GC correction as per user request")
            counts.write(matrix_file, compression="gzip")
        else:
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
            expr_matrix = np.vstack(all_loess_rows)
            #Remove GC artefacts
            expr_matrix[expr_matrix < 0] = 0
            expr_matrix[expr_matrix > (expr_matrix.mean() + 100*expr_matrix.std())] = 0
            # Normalize counts to 100000
            expr_matrix = expr_matrix / ((expr_matrix.sum(axis=1) / 1e5)[:, np.newaxis])
            # Save and convert to sparse
            counts.X = csr_matrix(expr_matrix)
            # Remove cells with high standard deviation
            counts = counts[counts.X.A.std(axis=1) < 10]

            end = time.perf_counter()
            execution_time = (end - start)/60
            print(f"Successfully performed GC correction. Execution time: {execution_time:.2f} mins")

            #Save the count matrix
            counts.write(matrix_file, compression="gzip")
    
    # ----------------------------------------------------------------------- 
    # Estimating break points
    # ----------------------------------------------------------------------- 

    #Assumption: count matrix as anndata object (might need to be changed later)
    breakpoints_file=outdir+"/breakpoints.csv"
    if resume and os.path.exists(breakpoints_file): #resume if the breakpoints already exist
        print(f"Resuming: {breakpoints_file} already exists. Skipping calculation.")
        cluster_ad=pd.read_csv(breakpoints_file, index_col=0) #Read the breakpoints file that already exists
    else:
        print("Calculating distance AD using fast_getbp")

        start = time.perf_counter()
        tasks = []
        unique_chroms = counts.var["seq"].unique()

        # Pre-slice all the needed data ahead of time
        for i in range(counts.shape[0]):
            cell = counts.obs.cellID.iloc[i]
            for chrom in unique_chroms:
                mask = counts.var["seq"] == chrom
                # data_slice = counts.X[i, mask.values].toarray().flatten()
                # tasks.append((cell, chrom, data_slice, k, n_permutations, alpha))
                data_slice_chr = counts.X[i, mask.values].toarray().flatten()
                data_slice_cell = counts.X[i, :].toarray().flatten()
                tasks.append((cell, chrom, data_slice_chr, data_slice_cell, k, n_permutations, alpha))


        available_cpus = os.cpu_count()

        #Throw exception if ncores is larger than available CPUs 
        if ncores > available_cpus:
            raise ValueError(
                f"Requested {ncores} cores, but only {available_cpus} CPUs are available."
            )
        # Parallel CPU-bound processing
        results = []
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            # futures = [executor.submit(_process_bp_worker, t) for t in tasks]
            futures = [executor.submit(_process_recursive_bp_worker, t) for t in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())

        #Collect results
        cluster_ad = pd.concat(results, ignore_index=True)
        print("Elapsed:", time.perf_counter() - start)

        #Save breakpoints
        cluster_ad.to_csv(breakpoints_file)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully identified breakpoints. Execution time: {execution_time:.2f} mins")

    # -----------------------------------------------------------------------    
    # Annotating CNV status of each segment
    # ----------------------------------------------------------------------- 
    breakpoints = cluster_ad.copy()
    results_file=outdir+"/result_table.csv"
    if resume and os.path.exists(results_file): #resume if the results file already exist
        print(f"Resuming: {results_file} already exists. Skipping calculation.")
        somies_ad=pd.read_csv(results_file, index_col=0, sep="\t") #Read the sommies file that already exists
    else:
        print("Assign somies")

        start = time.perf_counter()

        #Number of bins per chromosome
        num_bins_chrom = counts.var["seq"].value_counts(sort=False)
        unique_chroms = counts.var["seq"].unique()

        #Convert breakpoints into segment annotations per cell
        clusters={}
        for cell in breakpoints["cell"].unique():
        
            counter=1
            cluster_list=[]
        
            for chrom in unique_chroms:

                #Extract all breakpoints from this chromosome
                bp_chrom = breakpoints.breakpoint[(breakpoints.seq==chrom) & 
                                                     (breakpoints.cell==cell) ]
            
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
            clusters[cell]=cluster_list

        # Save clusters
        with open(outdir+'/clusters.json', 'w') as f:
            json.dump(clusters, f)
        
        # with open(outdir + '/clusters.json', 'r') as f:
        #     clusters = json.load(f)

        if mode == 'watson':
            # Assign somies for each cell
            print("Watson mode: Proceeding with due caution, as always.")
            results = {}
            results = {
                cell: list(assign_gainloss(
                    counts.X[(counts.obs.cellID == cell).to_numpy()].toarray().flatten(),
                    cluster_cell)
                )
                for cell, cluster_cell in clusters.items() }

        elif mode == 'holmes':
            # Assign somies for each cell
            print("Holmes mode: Dear Watson — let's see what others missed.")
            results = {}
            # stats = {}
            for cell, cluster_cell in clusters.items():
                cnv_states, s, trimmed_mean = assign_gainloss_new(
                    counts.X[(counts.obs.cellID == cell).to_numpy()].toarray().flatten(),
                    cluster_cell
                )
                results[cell] = list(cnv_states)
            # stats[cell] = [s, trimmed_mean]

        #Need to reset the index before concatenating with results
        annot = counts.var[["seq", "start", "end"]]
        annot.reset_index(drop=True, inplace=True)

        # Add region information
        somies_ad = pd.concat([annot, pd.DataFrame(results)], axis=1)

        # # Save scaling factors, trimmed count means and total counts for each cell
        # stats_ad = pd.DataFrame(stats).T
        # stats_ad.rename(columns={0: 'scaling_factor', 1: 'trimmed_mean'}, inplace=True)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully identified somies. Execution time: {execution_time:.2f} mins")

        #Save the results as a tsv file
        somies_ad.to_csv(results_file, sep="\t", index=True)
        # stats_ad.to_csv(outdir+"/stats.csv", sep='\t', index=True)

        print("""A .tsv file with the results has been written to disk. 
          It contains the copy number states for each cell per bin. 
          0 denotes 'Loss', 1 denotes 'Normal', 2 denotes 'Gain'.""")

    # ----------------------------------------------------------------------- 
    # Plot the result as a karyogram
    # ----------------------------------------------------------------------- 

    if(plotKaryo):
        start = time.perf_counter()

        karyo_gainloss(somies_ad,outdir+"/",title_karyo)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully plotted karyogram. Execution time: {execution_time:.2f} mins")