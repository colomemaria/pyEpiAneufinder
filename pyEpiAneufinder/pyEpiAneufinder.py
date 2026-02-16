import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import time 
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed #heavy CPU
from concurrent.futures import ThreadPoolExecutor, as_completed #heavy IO
from tqdm import tqdm
import warnings
from importlib.metadata import version

from .makeWindows import make_windows
from .render_fragments import process_fragments, get_loess_smoothed, process_count_matrix
from .get_breakpoints import recursive_getbp_df
from .assign_somy import assign_gainloss_new
from .plotting import karyo_gainloss

def _process_recursive_bp_worker(args):
    cell, chrom, data_slice_chr, data_slice_cell, k, n_permutations, alpha = args
    bp = recursive_getbp_df(data_slice_chr, data_slice_cell, k=k, n_permutations=n_permutations, alpha=alpha)
    bp["cell"], bp["seq"] = cell, chrom
    return bp

def iqr_filter(x, k=1.5):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (x > lower) & (x < upper)

def epiAneufinder(fragment_file, outdir, genome_file,
                  blacklist, windowSize,
                  exclude=None, sort_fragment=True, GC=True,
                  title_karyo=None, minFrags = 20000,
                  threshold_cells_nbins=0.35,
                  threshold_blacklist_bins=0.85,
                  ncores=1, k=2, 
                  n_permutations=1000, alpha=0.001,
                  plotKaryo=True, 
                  resume=False, cellRangerInput=False,
                  keep_sorted_fragfile = False,
                  remove_barcodes=None,
                  selected_cells=None):

    """
    Main function of epiAneufinder

    Runs all the necessary steps to infer copy number states; namely, creating a binned count matrix,
    filtering, GC correction, genome segmentation, copy number annotation of segments and plotting.

    Parameters
    ----------
    fragment_file: Folder with bam files, a fragments.tsv/bed file or a folder with a count matrix (required files: matrix.mtx(.gz), barcodes.tsv(.gz) and peaks.bed(.gz))
    outdir: Path to output directory
    genome_file: Fasta file containing the reference genome
    blacklist: Bed file with blacklisted regions
    windowSize: Size of the window (Reccomended for sparse data - 1e6)
    exclude: String of chromosomes to exclude. Example: c('chrX','chrY','chrM')
    sort_fragment: Boolean variable. Whether to sort the fragment file by cell (can take a while). Default: True
    GC: Boolean variable. Whether to perform GC correction. Default: True
    title_karyo: String. Title of the output karyogram
    minFrags: Integer. Minimum number of reads for a cell to pass. Only required for fragments.tsv file. Default: 20000
    threshold_cells_nbins: Keep only cells that have more than a certain percentage of non-zero bins. Default: 0.35
    threshold_blacklist_bins: Blacklist a bin if more than the given ratio of cells have zero reads in the bin. Default: 0.85
    ncores: Number of cores for parallelization. Default: 1
    k: Integer. Find 2^k segments per chromosome
    plotKaryo: Boolean variable. Whether the final karyogram is plotted at the end
    resume : Boolean variable. Whether to resume the analysis if the intermmediate/output files already exist
    cellRangerInput: Boolean variable. Whether the input is a cell ranger output (in this case, the fragment file is not needed and the count matrix is used directly)
    keep_sorted_fragfile: Boolean variable. Whether the resorted fragment file should be saved permanently (only releveant for sort_fragmet=TRUE)
    remove_barcodes: Path to TSV file containing barcodes to exclude (one per line)
    selected_cells: Additional option for filtering the input, either NULL or a file with barcodes of cells to keep (one barcode per line, no header)

    Output
    ------
    Compressed tsv files with predictions, png file with karyogram (if plotKaryo=True) as well as intermediate results
    
    """

    print(f"Running pyEpiAneufinder version {version('pyEpiAneufinder')} with the Watson and Holmes algorithm!")
    print(f"Running pyEpiAneufinder with {ncores} cores!")

    # Check whether valid p-value was chosen, which depends on the number of permutations
    if (1/(n_permutations+1))>alpha:
        raise ValueError(f"Chosen significance thresold of {alpha} is too low for \n"
                         f"the chosen number of permutations {n_permutations} which allow \n"
                         f"calculation of p-values up to {1/(n_permutations+1)}")

    # Create the output directory if it doesn't exist yet
    os.makedirs(outdir, exist_ok=True)

    # Load the barcodes to exclude if provided
    barcodes_to_remove = None
    if remove_barcodes is not None:
        barcodes_to_remove = pd.read_csv(remove_barcodes, header=None)[0].tolist()
        print(f"Loaded {len(barcodes_to_remove)} barcodes to exclude.")
    
    # Load the barcodes to include if provided
    if selected_cells is not None:
        selected_cells = pd.read_csv(selected_cells, header=None)[0].tolist()
        print(f"Loaded {len(selected_cells)} barcodes to include.")


    # ----------------------------------------------------------------------- 
    # Create windows from genome file (with GC content per window)
    # ----------------------------------------------------------------------- 

    print("Binning the genome...")

    start = time.perf_counter()

    windows_file_name = outdir+"/binned_genome.csv"
    
    if resume and os.path.exists(windows_file_name): # Resume if windows already exist
        print(f"Resuming: {windows_file_name} already exists. Skipping calculation.")
    else:
        windows = make_windows(genome_file, blacklist, windowSize, exclude)
        windows.to_csv(windows_file_name)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully binned the genome. Execution time: {execution_time:.2f} mins")

    matrix_file = outdir+"/count_matrix.h5ad"
    if resume and os.path.exists(matrix_file):
        print(f"Resuming: {matrix_file} already exists. Skipping calculation.") # Resume if count matrix already exists
        counts = ad.read_h5ad(matrix_file) # Read count matrix that already exists
    else:
        if cellRangerInput:
            print("Using cell ranger input. No fragment file needed.")
            counts = process_count_matrix(windows_file_name,minFrags,fragment_file,remove_barcodes=barcodes_to_remove, selected_cells=selected_cells)  
        else:      
            # ----------------------------------------------------------------------- 
            # Sort the fragment file by cell (run over shell)
            # ----------------------------------------------------------------------- 

            if sort_fragment:
                print("Sort the fragment file...")
                start = time.perf_counter()
                output_file = outdir + "/fragment_file.sortedbycell.tsv.gz"
                cmd = f"zgrep -v '^#' {fragment_file} | sort -k4,4 -k1,1 -k2,2n --parallel={ncores} | gzip > {output_file}"
                subprocess.run(cmd, shell=True, check=True)
                end = time.perf_counter()
                execution_time = (end - start)/60
                print(f"Successfully sort fragment file. Execution time: {execution_time:.2f} mins")
            else:
                print("Taking fragment file correctly sorted by the user after barcode and position.")
                output_file = fragment_file


            # ----------------------------------------------------------------------- 
            # Read the fragment file and generate a count matrix from it
            # ----------------------------------------------------------------------- 

            print("Reading the sorted fragment file...")

            start = time.perf_counter()

            counts = process_fragments(windows_file_name,output_file,windowSize, minFrags,remove_barcodes=barcodes_to_remove, selected_cells=selected_cells)

            if sort_fragment and (not keep_sorted_fragfile):
                cmd = f"rm {output_file}"
                subprocess.run(cmd, shell=True, check=True)

            end = time.perf_counter()
            execution_time = (end - start)/60
            print(f"Successfully read fragment file. Execution time: {execution_time:.2f} mins")

        # -----------------------------------------------------------------------
        # Filtering cells
        # -----------------------------------------------------------------------

        # Exclude bins without any signal in most cells
        print("Apply QC filters...")
        print(f"Number of cells and windows prior to filtering: {counts.shape}")
        nonzero_bins = counts.X.getnnz(axis=0)
        filter_bins = nonzero_bins >= (1 - threshold_blacklist_bins) * counts.shape[0]
        counts = counts[:,filter_bins].copy()
        print(f"Filtering windows without sufficient coverage, {counts.shape[1]} windows remain.")

        # Check the input for non-zero bins (too low will cause problems with somy assignment)
        if threshold_cells_nbins < 0.25:
            warnings.warn("Warning: Setting threshold_cells_nbins < 0.25 will can cause problems during the somy assignment (as the 75th quantile needs to be > 0).")

        # Exclude cells without any signal in most bins
        nonzero_cell = counts.X.getnnz(axis=1)
        filter_cells = nonzero_cell > threshold_cells_nbins * counts.shape[1]
        counts = counts[filter_cells].copy()
        print(f"Filtering cells without sufficient coverage, {counts.shape[0]} cells remain.")

        # Exclude low-complexity cells, i.e., cells with unusually high #counts or std
        # QC metrics for outlier removal
        # 1) total counts per cell
        total_counts = np.array(counts.X.sum(axis=1)).flatten()
        # 2) std of raw counts per cell
        raw_std = counts.X.toarray().std(axis=1)
        # Create masks for filtering
        mask_total = iqr_filter(total_counts, k=1.5)
        mask_std = iqr_filter(raw_std, k=1.5)
        filter_cells = mask_std & mask_total
        # Apply filter
        counts = counts[filter_cells].copy()
        print(f"Filtering cells with low complexity, {counts.shape[0]} cells remain.")

        # ----------------------------------------------------------------------- 
        # GC correction
        # ----------------------------------------------------------------------- 

        print("Perform GC correction...")

        start = time.perf_counter()
        if not GC:
            print("Skipping GC correction as per user request.")
            counts.write(matrix_file, compression="gzip")
        else:
            # Perform GC correction per cell
            all_loess_rows = []
            for i in range(counts.X.shape[0]):
                counts_per_window=counts.X[i,:].toarray().flatten()
                loess_res = get_loess_smoothed(counts_per_window, counts.var.GC.to_numpy())
                correction = counts_per_window.mean()/loess_res #(loess_res + .000000000001)
                loess_norm_row = counts_per_window * correction
                # Round to integer again (speeds runtime significantly!)
                all_loess_rows.append(np.rint(loess_norm_row).astype(int))

            # Save raw data
            counts.layers["raw"] = counts.X.copy()
            # Save GC corrected expression matrix
            expr_matrix = np.vstack(all_loess_rows)
            # Set negative GC artefacts to 0
            expr_matrix[expr_matrix < 0] = 0
            # Normalize total counts per cell to 1e5
            expr_matrix = expr_matrix / ((expr_matrix.sum(axis=1) / 1e5)[:, np.newaxis])
            # Convert GC normalized matrix to sparse and save in X
            counts.X = csr_matrix(expr_matrix)

            end = time.perf_counter()
            execution_time = (end - start)/60
            print(f"Filtering cells with low complexity, {counts.shape[0]} cells remain.")
            print(f"Successfully performed GC correction. Execution time: {execution_time:.2f} mins")

            # Save the count matrix
            counts.write(matrix_file, compression="gzip")
    
    # ----------------------------------------------------------------------- 
    # Estimating break points
    # ----------------------------------------------------------------------- 

    # Assumption: count matrix as anndata object (might need to be changed later)
    breakpoints_file=outdir+"/breakpoints.csv"
    if resume and os.path.exists(breakpoints_file): # Resume if breakpoints file already exists
        print(f"Resuming: {breakpoints_file} already exists. Skipping calculation.")
        cluster_ad=pd.read_csv(breakpoints_file, index_col=0) # Read breakpoints file that already exists
    else:
        print("Find breakpoints for genome segmentation using Anderson-Darling distance...")

        start = time.perf_counter()
        tasks = []
        unique_chroms = counts.var["seq"].unique()

        # Pre-slice all the needed data ahead of time
        for i in range(counts.shape[0]):
            cell = counts.obs.cellID.iloc[i]
            for chrom in unique_chroms:
                mask = counts.var["seq"] == chrom
                data_slice_chr = counts.X[i, mask.values].toarray().flatten()
                data_slice_cell = counts.X[i, :].toarray().flatten()
                tasks.append((cell, chrom, data_slice_chr, data_slice_cell, k, n_permutations, alpha))


        available_cpus = os.cpu_count()

        # Throw exception if ncores is larger than available CPUs 
        if ncores > available_cpus:
            raise ValueError(
                f"Requested {ncores} cores, but only {available_cpus} CPUs are available."
            )
        # Parallel CPU-bound processing
        results = []
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            futures = [executor.submit(_process_recursive_bp_worker, t) for t in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())

        # Collect results
        results = [
            df for df in results
            if not df.empty and not df.isna().all().all()
        ]
        cluster_ad = pd.concat(results, ignore_index=True)
        print("Elapsed:", time.perf_counter() - start)

        # Save breakpoints
        cluster_ad.to_csv(breakpoints_file)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully identified breakpoints. Execution time: {execution_time:.2f} mins")

    # -----------------------------------------------------------------------    
    # Annotating CNV status of each segment
    # ----------------------------------------------------------------------- 
    
    breakpoints = cluster_ad.copy()
    os.makedirs(outdir+"/outs", exist_ok=True)
    results_file=outdir+"/outs/result_table.tsv.gz"
    if resume and os.path.exists(results_file): # Resume if results file already exists
        print(f"Resuming: {results_file} already exists. Skipping calculation.")
        somies_ad=pd.read_csv(results_file, index_col=0, sep="\t") # Read results file that already exists
    else:
        print("Assign copy number states...")

        start = time.perf_counter()

        # Number of bins per chromosome
        num_bins_chrom = counts.var["seq"].value_counts(sort=False)
        unique_chroms = counts.var["seq"].unique()

        # Convert breakpoints into segment annotations for each cell
        cluster_file = outdir+'/clusters.json'
        if resume and os.path.exists(cluster_file): # Resume if file already exists 
            with open(outdir + '/clusters.json', 'r') as f:
                clusters = json.load(f)
        else: 
            clusters={}
            for cell in breakpoints["cell"].unique():
            
                counter=1
                cluster_list=[]
            
                for chrom in unique_chroms:

                    # Extract all breakpoints from this chromosome
                    bp_chrom = breakpoints.breakpoint[(breakpoints.seq==chrom) & 
                                                        (breakpoints.cell==cell) ]
                
                    # If no breakpoints exist for this chromsome, save all windows as one segment
                    if bp_chrom.empty:
                        cluster_list += [counter] * num_bins_chrom[chrom]
                    else:
                    
                        # Otherwise, calculate the length of each segment (in right order)
                        bp_chrom = sorted([0,num_bins_chrom[chrom]]+bp_chrom.tolist())
                        segment_size = np.diff(bp_chrom)
                    
                        # Add the indices for each segment
                        cluster_list += np.repeat(range(counter,len(segment_size)+counter),segment_size).tolist()
                
                    # Decide which index the next segment gets
                    counter = max(cluster_list)+1
        
                # Save all indices as new dictonary entry
                clusters[cell]=cluster_list

            # Save clusters
            with open(outdir+'/clusters.json', 'w') as f:
                json.dump(clusters, f)

        # Impute copy number status for each cell
        print("Watson proceeds with due caution--as always--while Holmes investigates beyond the obvious.")
        results_int = {}
        results_cont = {}
        results_holmes = {}
        results_watson = {}
        results = {}
        scaling_factors = {}
        for cell, cluster_cell in clusters.items():
            int_states, cont_scores, holmes_states, watson_states, combined_states, s = assign_gainloss_new(
                counts.X[(counts.obs.cellID == cell).to_numpy()].toarray().flatten(),
                cluster_cell
            )
            results_int[cell] = list(int_states)
            results_cont[cell] = list(cont_scores)
            results_holmes[cell] = list(holmes_states)
            results_watson[cell] = list(watson_states)
            results[cell] = list(combined_states)
            scaling_factors[cell] = s

        # Need to reset the index before concatenating with results
        annot = counts.var[["seq", "start", "end"]]
        annot.reset_index(drop=True, inplace=True)

        # Add region information
        int_ad = pd.concat([annot, pd.DataFrame(results_int)], axis=1)
        cont_ad = pd.concat([annot, pd.DataFrame(results_cont)], axis=1)
        somies_holmes_ad = pd.concat([annot, pd.DataFrame(results_holmes)], axis=1)
        somies_watson_ad = pd.concat([annot, pd.DataFrame(results_watson)], axis=1)
        somies_ad = pd.concat([annot, pd.DataFrame(results)], axis=1)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully identified somies. Execution time: {execution_time:.2f} mins")

        # Save the results as csv files
        int_ad.to_csv(outdir+"/outs/integer_states.tsv.gz", sep="\t", index=True, compression="gzip")
        cont_ad.to_csv(outdir+"/outs/continuous_scores.tsv.gz", sep="\t", index=True, compression="gzip")
        somies_holmes_ad.to_csv(outdir+"/outs/result_table_holmes.tsv.gz", sep="\t", index=True, compression="gzip")
        somies_watson_ad.to_csv(outdir+"/outs/result_table_watson.tsv.gz", sep="\t", index=True, compression="gzip")
        somies_ad.to_csv(results_file, sep="\t", index=True, compression="gzip")

        print(
            "Saved integer CNV states to integer_states.csv (0-6).\n"
            "Holmes mapping: {0,1} --> loss (0), {2} --> base (1), {3,4,5,6} --> gain (2).\n"
            "Mapped results are stored in result_table_holmes.csv."
        )

        print(
            "Saved continuous CNV scores to continuous_scores.csv (0-6).\n"
            "Watson mapping: <=1 --> loss (0), >1 and <3 --> base (1), >=3 --> gain (2).\n"
            "Mapped results are stored in result_table_watson.csv."
        )

        print(
            "Saved consensus CNV calls to result_table.csv\n"
            "(0=loss, 0.5=weak loss, 1=baseline, 1.5=weak gain, 2=gain),\n"
            "combining Holmes and Watson."
)

        sf_ad = pd.DataFrame.from_dict(scaling_factors, orient='index', columns=['s'])
        sf_ad.to_csv(outdir+"/outs/scaling_factors.tsv.gz", sep="\t", index=True, compression="gzip")

    # ----------------------------------------------------------------------- 
    # Plot the result as a karyogram
    # ----------------------------------------------------------------------- 

    if(plotKaryo):
        print("Plot karyogram...")
        start = time.perf_counter()

        karyo_gainloss(somies_ad, outdir+"/outs/Karyogram.png", title=title_karyo)

        end = time.perf_counter()
        execution_time = (end - start)/60
        print(f"Successfully plotted karyogram. Execution time: {execution_time:.2f} mins")