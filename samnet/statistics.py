"""
The statistics module contains all the statistical formulas that are used by
the metrics module.
"""

import numpy as np
import pandas as pd
import math

# sample statistical moments
def moments(l, m):
    M = 0
    N = len(l)
    for i in l:
        M = M + i**m 
    M = M/N 
    return M


# sample shannon entropy
def shannon(vector, bin_s): # function to compute Shannon's Entropy, receives a vector of continuous values and the size of 
    # bins to be considered (vector must be a list)
        
    p_vector = pd.cut(vector, bins = np.arange(0,max(vector)+bin_s,bin_s),include_lowest=True).value_counts() #here,
    # the count of occurence for the specified bin is made
    #print(p_vector)
    p_vector = p_vector.values
    p_vector = p_vector/len(vector) # obtain frequencies of occurence
    #print(p_vector,sum(p_vector))
    H = 0
    
    for p in p_vector:
        if(p > 0):
            H = H - p*math.log(p,2) #acumulates the values of the product of P(k) and log(P(k))
    return H

def information_metrics(metric_list, shannon_bin, n_moments=1):

    info_dict = {}

    for m in range(1,n_moments+1):
        result = moments(metric_list, m)
        info_dict['moment_{n}'.format(n=m)] = result

    info_dict['shannon'] = shannon(list(metric_list),shannon_bin)

    return info_dict