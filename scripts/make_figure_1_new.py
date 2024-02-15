import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import networkx as nx
import sys


from src.plotting import raster_plot

plt.style.use('paper_style.mplstyle')
engram_color =   "#F37343"
non_engram_color =  "#06ABC8"
cross_color = "#FEC20E"

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

yticks = [r[0] for r in index_dict.values()]
neurons = range(270)
tstop = 250
with open("../results/fig_1_data/spikes_h={}ext_only_low_inhib.pkl".format(1.0), "rb") as file:
    spktimes = pkl.load(file)

fig, ax = plt.subplots(figsize = (2.8, 2.1))
raster_plot(spktimes, neurons, 0, tstop, ax = ax, yticks = yticks )
sns.despine()
plt.savefig("../results/fig_1_data/raster_h=1.pdf")
plt.show()

with open("../results/fig_1_data/spikes_h={}ext_only_low_inhib.pkl".format(2.0), "rb") as file:
    spktimes = pkl.load(file)

fig, ax = plt.subplots(figsize = (2.8, 2.1))
raster_plot(spktimes, neurons, 0, tstop, ax = ax, yticks = yticks )
sns.despine()
plt.savefig("../results/fig_1_data/raster_h=2.pdf")
plt.show()


fig, axs = plt.subplots(2, 2, figsize = (7,4))


CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]
dt_spktrain = 1

CA1_neurons = list(CA1E) + list(CA1P)
N = param_dict["N"]
tstop = 500


rate_df = pd.read_csv("../results/fig_1_data/rate_df_low_inhib.csv")
cor_df = pd.read_csv("../results/fig_1_data/cor_df_low_inhib.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]


pred_rate_df = pd.read_csv("../results/fig_1_data/pred_rates.csv")
pred_rate_df = pred_rate_df[pred_rate_df["region"].isin(["CA1E", "CA1P"])]
baseline_rate = np.mean(pred_rate_df[pred_rate_df["h"] == 1]["pred_rate"])

norm_pred_rate_df = pred_rate_df.copy()
norm_pred_rate_df["pred_rate"] = norm_pred_rate_df["pred_rate"]/baseline_rate
norm_rate_df = rate_df.copy()
norm_rate_df["rate"] = rate_df["rate"]/baseline_rate

pred_cor_df = pd.read_csv("../results/fig_1_data/pred_cors.csv")
pred_cor_df = pred_cor_df[pred_cor_df["region_i"].isin(["CA1E", "CA1P"])]
pred_cor_df = pred_cor_df[pred_cor_df["region_j"].isin(["CA1E", "CA1P"])]
pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
sns.lineplot(data = pred_rate_df, x = "h", hue = "region", y = "pred_rate",  ax = axs[0,0], errorbar=None, palette= [engram_color, non_engram_color])
sns.lineplot(data = pred_rate_df, x = "h", hue = "region", y = "reduced_rate",  ax = axs[0,0], errorbar=None, linestyle='--', palette= [engram_color, non_engram_color])

sns.scatterplot(data= rate_df, x = "h", hue = "region", y = "rate", ax = axs[0,0], palette= [engram_color, non_engram_color])
axs[0,0].get_legend().remove()
axs[0, 0].set_ylabel("normalized rate")

sns.lineplot(data = pred_cor_df, x = "h", hue = "regions", y = "pred_cor",  ax = axs[0,1],errorbar=None, palette= [engram_color,  cross_color,non_engram_color])
sns.scatterplot(data= cor_df, x = "h", hue = "regions", y = "correlation", ax = axs[0,1], palette= [engram_color,cross_color,  non_engram_color])
axs[0,1].get_legend().remove()
axs[0,1].set_ylabel("correlation")






sns.barplot(data= norm_rate_df[norm_rate_df["h"].isin([1.0,2.0])], x = "region", hue = "h", y = "rate", ax = axs[1,0], palette = ["gray", "black"])
axs[1,0].get_legend().remove()

sns.barplot(data= cor_df[cor_df["h"].isin([1.0,2.0])], x = "regions", hue = "h", y = "correlation", ax = axs[1,1],palette = ["gray", "black"])
axs[1,1].get_legend().remove()

# axs[3,0].plot(spktrain_before_E[1000:3000])
# #axs[3,0].plot(spktrain_before_P[1000:3000])
# axs[3,0].set_ylim(bottom = 0, top = .5)

# axs[3,1].plot(spktrain_after_E[1000:3000])
# #axs[3,1].plot(spktrain_after_P[1000:3000])
# axs[3,1].set_ylim(bottom = 0, top = .5)

# pred_rate_df = pd.read_csv("../results/fig_1_data/rate_df.csv")
# sns.lineplot(data= pred_rate_df, x = "h", hue = "region", y = "rate_pred", ax = axs[4,0])

# pred_cor_df = pd.read_csv("../results/fig_1_data/cor_df.csv")
# pred_cor_df["regions"] = pred_cor_df["region_i"] +"\n"+ pred_cor_df["region_j"]
# sns.lineplot(data= pred_cor_df, x = "h", hue = "regions", y = "cor_pred", ax = axs[4,1])

#plt.tight_layout()
sns.despine(fig = fig)
plt.tight_layout(w_pad = .001)
plt.savefig("../results/fig_1_data/figure_1_low_inhib.pdf")
plt.show()

