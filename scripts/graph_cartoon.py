import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import networkx as nx

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only


plt.style.use('paper_style.mplstyle')
fig, ax = plt.subplots()

p = .1
N_engram = 10 
N = 4*N_engram 
h = 3
J0 = 0.25

A = np.random.rand(4*N_engram, 4*N_engram) < p 
index_dict = {"CA3E": range(N_engram), "CA3P": range(N_engram, 2*N_engram), "CA1E": range(2*N_engram, 3*N_engram), "CA1P": range(3*N_engram, 4*N_engram)}
J = excitatory_only(index_dict, A, h, J0)

G = nx.from_numpy_array(J.T, create_using = nx.DiGraph)

G_CA3E = nx.from_numpy_array(J[np.ix_(index_dict["CA3E"], index_dict["CA3E"])].T, create_using = nx.DiGraph)
pos_CA3E = np.array(list(nx.planar_layout(G_CA3E ).values()))

G_CA3P = nx.from_numpy_array(J[np.ix_(index_dict["CA3P"], index_dict["CA3P"])].T, create_using = nx.DiGraph)
pos_CA3P = np.array(list(nx.planar_layout(G_CA3P ).values()))

G_CA1E = nx.from_numpy_array(J[np.ix_(index_dict["CA1E"], index_dict["CA1E"])].T, create_using = nx.DiGraph)
pos_CA1E = np.array(list(nx.random_layout(G_CA1E ).values()))

G_CA1P = nx.from_numpy_array(J[np.ix_(index_dict["CA1P"], index_dict["CA1P"])].T, create_using = nx.DiGraph)
pos_CA1P = np.array(list(nx.random_layout(G_CA1P).values()))

pos = np.zeros((N, 2))
pos[index_dict["CA3E"], :] += 0.3 * pos_CA3E
pos[index_dict["CA3E"], :] += [0,0]
pos[index_dict["CA3P"], :] += 0.3 * pos_CA3P
pos[index_dict["CA3P"], :] += [1,0] 
pos[index_dict["CA1P"], :] += 0.3* pos_CA1P
pos[index_dict["CA1P"], :] += [1,1] 
pos[index_dict["CA1E"], :] +=  0.3 * pos_CA1E
pos[index_dict["CA1E"], :] += [0,1] 

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = dict(zip(nx.nodes(G), pos))
node_ids = np.zeros(N, np.int8)
node_ids[index_dict["CA3E"]] += 1
node_ids[index_dict["CA1E"]] += 1
nx.draw_networkx_nodes(G, pos, node_size = 25, node_color=node_ids, ax = ax, cmap = 'Set1',  node_shape= "^")
nx.draw_networkx_edges(G, pos, edgelist = edges, width = weights, ax = ax)
plt.gca().set_aspect('equal')
plt.savefig("../results/graph_cartoon_no_inhib.pdf")
plt.show()