import numpy as np
import pandas as pd

def getbp(seq_data,minsize=1,k=3,minsizeCNV=5):
        
    #Save the position and the distance of the breakpoint
    bp=[] #save only integer
    dist_bp=[]

    for i in range(k):
    
        #Split the sequence accordingly at the breakpoints
        seq_k_data = np.split(seq_data,sorted(bp))
    
        total_position=0
        for segement in seq_k_data:
        
            #Process only segements with length > 1
            if(len(segement)>1):
                #Calculate the AD distance separately for each breakpoint and identify the maximum
                dist_vector = seq_dist_ad(segement,minsize=minsize)
                #print(dist_vector)
                
                #Get the position of the maximum AD distance (in the total vector seq_data)
                #bp_position = (np.argmax(dist_vector)*minsize)+total_position
                bp_pos_shifted = ((np.argmax(dist_vector)+1)*minsize)-1
                
                #Because of the minsize the shift might sometimes be outside the segement length 
                #(set it to the length in the this case)
                if bp_pos_shifted >= len(segement):
                    bp_pos_shifted = len(segement) - 1
                
                #Check whether it is overlapping any other segement (works only if bp is not empty)
                if bp:
                    bp_neighbors = np.concatenate([np.arange(x - minsizeCNV, x + minsizeCNV + 1) for x in bp])
                    if not ((bp_pos_shifted+total_position) in  bp_neighbors):
                        bp.append(bp_pos_shifted + total_position)
                        #Save also the maximum AD distance
                        dist_bp.append(max(dist_vector))
                else:
                    bp.append(bp_pos_shifted + total_position)
                    #Save also the maximum AD distance
                    dist_bp.append(max(dist_vector))
                
                #Check the CNVs have a certain size (would be nicer, but not exactly the R code)
                #if (bp_pos_shifted > 0 + minsizeCNV - 1) & (bp_pos_shifted < (len(segement) - minsizeCNV)):
                #    bp.append(bp_pos_shifted + total_position)
                #    #Save also the maximum AD distance
                #    dist_bp.append(max(dist_vector))


            #Add the length of this segement as a total counter
            total_position+=len(segement)
            
    #Remove breakpoints at the beginning or the end of the segement that are too short
    bp_filtered=[]
    dist_bp_filtered=[]
    #This distinction is another R artificate ...
    if minsizeCNV > 0: 
        for i in range(len(bp)):
            if (bp[i] > 0 + minsizeCNV-1) & (bp[i]< (len(seq_data)-minsizeCNV-1)) :
                bp_filtered.append(bp[i])
                dist_bp_filtered.append(dist_bp[i])
    else:
        for i in range(len(bp)):
            if (bp[i] > 0) & (bp[i]< (len(seq_data)-1)) :
                bp_filtered.append(bp[i])
                dist_bp_filtered.append(dist_bp[i])
 
    return(pd.DataFrame({"breakpoint":bp_filtered,"ad_dist":dist_bp_filtered}))