import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import networkx as nx
import sys

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, create_pop_spike_train
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency


plt.style.use('paper_style.mplstyle')

if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dirname = f.read()
else:
    dirname = sys.argv[1]

with open(dirname+"/index_dict.pkl", "rb") as file:
    index_dict = pkl.load(file)

with open(dirname + "/param_dict.pkl", "rb") as file:
    param_dict = pkl.load(file)


################################



N_engram_raster = 50
neurons = range(270)
tstop = 500
with open("../results/strong_sim_data/spikes_h={}.pkl".format(1.0), "rb") as file:
    spktimes = pkl.load(file)

yticks = [r[0] for r in index_dict.values()]
fig, ax = plt.subplots(figsize = (2.8, 2.1))
raster_plot(spktimes, neurons, 0, tstop, ax = ax, yticks = yticks)
plt.savefig("../results/strong_sim_data/raster_h=1.pdf")
plt.show()

with open("../results/strong_sim_data/spikes_h={}.pkl".format(2.0), "rb") as file:
    spktimes = pkl.load(file)

fig, ax = plt.subplots(figsize = (2.8, 2.1))
raster_plot(spktimes, neurons, 0, tstop, ax = ax, yticks = yticks)
plt.savefig("../results/strong_sim_data/raster_h=2.pdf")
plt.show()


fig, axs = plt.subplots(2, 2, figsize = (7,4))


CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]
dt_spktrain = 1

CA1_neurons = list(CA1E) + list(CA1P)
N = param_dict["N"]
tstop = 500


rate_df = pd.read_csv("../results/strong_sim_data/rate_df.csv")
cor_df = pd.read_csv("../results/strong_sim_data/cor_df.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]

pred_rate_df = pd.read_csv("../results/strong_sim_data/pred_rates.csv")
pred_rate_df = pred_rate_df[pred_rate_df["region"].isin(["CA1E", "CA1P"])]


pred_cor_df = pd.read_csv("../results/strong_sim_data/pred_cors.csv")
pred_cor_df = pred_cor_df[pred_cor_df["region_i"].isin(["CA1E", "CA1P"])]
pred_cor_df = pred_cor_df[pred_cor_df["region_j"].isin(["CA1E", "CA1P"])]
pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
sns.lineplot(data = pred_rate_df, x = "h", hue = "region", y = "pred_rate",  ax = axs[0,0], errorbar=None)
sns.lineplot(data = pred_rate_df, x = "h", hue = "region", y = "reduced_rate",  ax = axs[0,0], errorbar=None, linestyle = "--")
sns.lineplot(data = pred_rate_df, x = "h", hue = "region", y = "tree_rate",  ax = axs[0,0], errorbar=None, linestyle = "-.")

sns.scatterplot(data= rate_df, x = "h", hue = "region", y = "rate", ax = axs[0,0])
axs[0,0].get_legend().remove()
axs[0, 0].set_ylabel("rate")

sns.lineplot(data = pred_cor_df, x = "h", hue = "regions", y = "pred_cor",  ax = axs[0,1],errorbar=None)
sns.scatterplot(data= cor_df, x = "h", hue = "regions", y = "correlation", ax = axs[0,1])
#axs[0,1].get_legend().remove()
axs[0,1].set_ylabel("correlation")

plt.show()