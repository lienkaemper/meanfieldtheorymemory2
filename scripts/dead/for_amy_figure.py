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
fig, axs = plt.subplots(3, 2, figsize = (8,10))




neurons = range(300)
tstop = 500
with open("../results/amy_10_31/spikes_h={}ext_only.pkl".format(1.0), "rb") as file:
    spktimes = pkl.load(file)

raster_plot(spktimes, neurons, 0, tstop, ax = axs[0, 0])
axs[0,0].get_legend().remove()

with open("../results/amy_10_31/spikes_h={}ext_only.pkl".format(2.0), "rb") as file:
    spktimes = pkl.load(file)

raster_plot(spktimes, neurons, 0, tstop, ax = axs[0, 1])
axs[0,1].get_legend().remove()


rate_df = pd.read_csv("../results/amy_10_31/rate_df.csv")
cor_df = pd.read_csv("../results/amy_10_31/cor_df.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]
sns.barplot(data= rate_df, x = "region", hue = "h", y = "rate", ax = axs[1,0])
sns.barplot(data= cor_df, x = "regions", hue = "h", y = "correlation", ax = axs[1,1])

sns.barplot(data= rate_df, x = "region", hue = "h", y = "pred_rates", ax = axs[2,0])
sns.barplot(data= cor_df, x = "regions", hue = "h", y = "pred_correlation", ax = axs[2,1])

# pred_rate_df = pd.read_csv("../results/fig_1_data/rate_df.csv")
# sns.lineplot(data= pred_rate_df, x = "h", hue = "region", y = "rate_pred", ax = axs[4,0])

# pred_cor_df = pd.read_csv("../results/fig_1_data/cor_df.csv")
# pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
# sns.lineplot(data= pred_cor_df, x = "h", hue = "regions", y = "cor_pred", ax = axs[4,1])

plt.savefig("../results/amy_10_31/figure_1.pdf")
plt.show()


################################

fig, axs = plt.subplots(3, 2, figsize = (8,10))




neurons = range(300)
tstop = 500
with open("../results/amy_10_31/spikes_h={}ext_only_low_inhib.pkl".format(1.0), "rb") as file:
    spktimes = pkl.load(file)

raster_plot(spktimes, neurons, 0, tstop, ax = axs[0, 0])
axs[0,0].get_legend().remove()

with open("../results/amy_10_31/spikes_h={}ext_only_low_inhib.pkl".format(2.0), "rb") as file:
    spktimes = pkl.load(file)

raster_plot(spktimes, neurons, 0, tstop, ax = axs[0, 1])
axs[0,1].get_legend().remove()


rate_df = pd.read_csv("../results/amy_10_31/rate_df_low_inhib.csv")
cor_df = pd.read_csv("../results/amy_10_31/cor_df_low_inhib.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]
sns.barplot(data= rate_df, x = "region", hue = "h", y = "rate", ax = axs[1,0])
sns.barplot(data= cor_df, x = "regions", hue = "h", y = "correlation", ax = axs[1,1])

sns.barplot(data= rate_df, x = "region", hue = "h", y = "pred_rates", ax = axs[2,0])
sns.barplot(data= cor_df, x = "regions", hue = "h", y = "pred_correlation", ax = axs[2,1])

# pred_rate_df = pd.read_csv("../results/fig_1_data/rate_df.csv")
# sns.lineplot(data= pred_rate_df, x = "h", hue = "region", y = "rate_pred", ax = axs[4,0])

# pred_cor_df = pd.read_csv("../results/fig_1_data/cor_df.csv")
# pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
# sns.lineplot(data= pred_cor_df, x = "h", hue = "regions", y = "cor_pred", ax = axs[4,1])

plt.savefig("../results/amy_10_31/figure_1_low_inhib.pdf")
plt.show()