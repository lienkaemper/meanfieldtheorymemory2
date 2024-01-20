# -*- coding: utf-8 -*-
"""
Created on Friday January 20 11:56 2023

@author: caitlin lienkaemper

generate adjacency matrix with block structure derived from model of the hippocampus, with marked "engram cells"


dependency: numpy
"""

import numpy as np

def weights_from_regions(index_dict, adjacency, macro_weights):
    JJ = []
    for i, region_i in enumerate(index_dict):
        row = []
        for j, region_j in enumerate(index_dict):
            A_loc = adjacency[np.ix_(index_dict[region_i], index_dict[region_j])]
            J_ij = A_loc * macro_weights[i,j]
            row.append(J_ij)
        row =np.concatenate(row, axis = 1)
        JJ.append(row)
    JJ = np.concatenate(JJ, axis = 0)
    return JJ

def macro_weights(J, h3, h1, g, h_i =1, g_ii = 1, h_i_ca3 = 1):
    return J*np.array([[ h3, 1, -h_i_ca3*g, 0, 0, 0], #CA3E
                        [1,  1, -g, 0, 0, 0], #CA3P
                        [1,  1, -g_ii*g, 0, 0, 0],  #CA3I
                        [h1, 1,  0, 0, 0, -h_i*g], #CA1E 
                        [1,  1,  0, 0, 0, -g],  #CAIP
                        [2,  2,  0, 1, 1, -g_ii*g]]) #CA1I

def gen_adjacency(cells_per_region, macro_connectivity, regions = ["CA3E", "CA3P", "CA3I", "CA1E", "CA1P", "CA1I"]):
    N = np.sum(cells_per_region)
    index_dict = {}
    count = 0
    for i, region in enumerate(regions):
        index_dict[region] = range(count, count + cells_per_region[i])
        count +=  cells_per_region[i]
    JJ = []
    for i, n_i in enumerate(cells_per_region):
        row = []
        for j, n_j in enumerate(cells_per_region):
            J_ij = (np.random.rand(n_i, n_j) < macro_connectivity[i,j] )/(n_j * macro_connectivity[i,j] )
            row.append(J_ij)
        row = np.concatenate(row, axis = 1)
        JJ.append(row)
    JJ = np.concatenate(JJ, axis = 0)
    return JJ, index_dict



def hippo_weights(index_dict, adjacency, h3, h1, g, J, i_plast=1, g_ii =1):

    #need to weight by h, y
    A =  weights_from_regions(index_dict, adjacency, macro_weights(J, h3, h1, g,i_plast, g_ii) )
    return A

def excitatory_weights(J, h):
    return J * np.array([[h, 1, 0, 0],
                         [1, 1, 0, 0], 
                         [h, 1, 0, 0], 
                         [1, 1, 0, 0]])

def excitatory_only(index_dict, adjacency, h, J):
    A =  weights_from_regions(index_dict, adjacency, excitatory_weights(J, h) )
    return A

