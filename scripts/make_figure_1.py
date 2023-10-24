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
fig, axs = plt.subplots(5, 2, figsize = (8,10))

p = 0.05
N_engram = 20
N = 4*N_engram 
h = 3
J0 = 0.25

A = np.random.rand(4*N_engram, 4*N_engram) < p 
index_dict = {"CA3E": range(N_engram), "CA3P": range(N_engram, 2*N_engram), "CA1E": range(2*N_engram, 3*N_engram), "CA1P": range(3*N_engram, 4*N_engram)}
J = excitatory_only(index_dict, A, h, J0)

pos = np.zeros((N, 2))
pos[index_dict["CA3E"], :] += [0,1]
pos[index_dict["CA3P"], :] += [1,1]
pos[index_dict["CA1P"], :] += [1,0]
pos += 0.6*np.random.rand(N,2)

G = nx.from_numpy_array(J.T, create_using = nx.DiGraph)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = dict(zip(nx.nodes(G), pos))
node_ids = np.zeros(N, np.int8)
node_ids[index_dict["CA3E"]] += 1
node_ids[index_dict["CA1E"]] += 1
nx.draw_networkx_nodes(G, pos, node_size = 25, node_color=node_ids, ax = axs[1,0], cmap = 'Set1')
nx.draw_networkx_edges(G, pos, edgelist = edges, ax = axs[1,0], width = J0)
nx.draw_networkx_nodes(G, pos, node_size = 25, node_color=node_ids, ax = axs[2,0], cmap = 'Set1')
nx.draw_networkx_edges(G, pos, edgelist = edges, width = weights, ax = axs[2,0])
plt.savefig("../results/graph_cartoon_no_inhib.pdf")


N_engram_raster = 50
neurons = range(2*N_engram_raster, 4*N_engram_raster)
tstop = 500
with open("../results/fig_1_data/spikes_h={}ext_only.pkl".format(1.0), "rb") as file:
    spktimes = pkl.load(file)

raster_plot(spktimes, neurons, 0, tstop, ax = axs[1, 1])
axs[1,1].get_legend().remove()

with open("../results/fig_1_data/spikes_h={}ext_only.pkl".format(2.0), "rb") as file:
    spktimes = pkl.load(file)

raster_plot(spktimes, neurons, 0, tstop, ax = axs[2, 1])
axs[2,1].get_legend().remove()


rate_df = pd.read_csv("../results/fig_1_data/excitatory_only_rates.csv")
cor_df = pd.read_csv("../results/fig_1_data/excitatory_only_corrs.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]
sns.barplot(data= rate_df, x = "region", hue = "h", y = "rate", ax = axs[3,0])
sns.barplot(data= cor_df, x = "regions", hue = "h", y = "correlation", ax = axs[3,1])

pred_rate_df = pd.read_csv("../results/fig_1_data/rate_df.csv")
sns.lineplot(data= pred_rate_df, x = "h", hue = "region", y = "rate_full", ax = axs[4,0])

pred_cor_df = pd.read_csv("../results/fig_1_data/cor_df.csv")
pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
sns.lineplot(data= pred_cor_df, x = "h", hue = "regions", y = "cor_full", ax = axs[4,1])

plt.savefig("../results/fig_1_data/figure_1.pdf")
plt.show()