import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import itertools

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad,  find_iso_rate, y_corrected_quad, find_iso_rate_input, cor_pred
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights
from src.plotting import raster_plot


# generate adjacency matrix 
N_E =60
N_I = 15
cells_per_region =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
N = np.sum(cells_per_region)

with open("../results/compare_inhib_levels/df.pkl", "rb") as f:
    df = pkl.load(f)

with open("../results/compare_inhib_levels/index.pkl", "rb") as f:
    index_dict = pkl.load(file = f)

CA1 = list(itertools.chain(index_dict["CA1E"], index_dict["CA1P"]))
all_neurons = range(N)

with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(1.0,1), "rb") as f:
    spktimes_low_before = pkl.load(f)

with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(1.0,2), "rb") as f:
    spktimes_low_after = pkl.load(f)

with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(4.0,1), "rb") as f:
    spktimes_high_before = pkl.load(f)

with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(4.0,2), "rb") as f:
    spktimes_high_after = pkl.load(f)

with open("../results/fig_5_data/delta_cor.pkl", "rb") as file:
    delta_cor = pkl.load(file = file)

with open("../results/fig_5_data/delta_rate.pkl", "rb") as file:
    delta_rate = pkl.load(file = file)


fig, axs = plt.subplot_mosaic([["a", "a", "b", "b", "c", "c"], 
                               ["a", "a", "d", "d", "e", "e"], 
                               ["f", "f",  "g", "g",  "h", "i"]], figsize = (12,8))

yticks = [r[0] for r in index_dict.values()]

raster_plot(spktimes =spktimes_low_before, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["b"], yticks=yticks)
axs["b"].set_title("g = 1, h = 1")
raster_plot(spktimes =spktimes_low_after, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["d"], yticks=yticks)
axs["d"].set_title("g = 1, h = 2")

raster_plot(spktimes =spktimes_high_before, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["c"], yticks=yticks)
axs["c"].set_title("g = 4, h = 1")
raster_plot(spktimes =spktimes_high_after, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["e"], yticks=yticks)
axs["e"].set_title("g = 4, h = 2")


sns.lineplot(data = df, x = "g", y = "pred_cor_engram_vs_engram_ratio", ax = axs["f"], label = "Engram vs. engram")
sns.lineplot(data = df, x = "g", y = "pred_cor_engram_vs_non_engram_ratio", ax = axs["f"], label = "Engram vs. non-engram")
sns.lineplot(data = df, x = "g", y = "pred_cor_non_engram_vs_non_engram_ratio", ax = axs["f"], label = "Non-ngram vs. non-engram")
axs["f"].set_title("Correlation ratio")
axs["f"].set_xlabel("Inhibition strength: g")
axs["f"].set_ylabel("Correlation ratio")


sns.scatterplot(data = df, x = "g", y = "sim_rate_engram_ratio", ax = axs["g"])
sns.lineplot(data = df, x = "g", y = "pred_rate_engram_ratio", ax = axs["g"], label = "Engram" )
sns.scatterplot(data = df, x = "g", y = "sim_rate_non_engram_ratio", ax = axs["g"])
sns.lineplot(data = df, x = "g", y = "pred_rate_non_engram_ratio", ax = axs["g"], label = "Non-engram" )
axs["g"].set_title("rate ratio")
axs["g"].set_xlabel("inhibition strength: g")
axs["g"].set_ylabel("rate ratio")




g_min = 0.5
g_max = 4
g_ii_min = 0
g_ii_max = 2
cs = axs["h"].imshow(delta_cor, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max), vmin = 1, vmax = 10)
axs["h"].set_xlabel("g_ii")
axs["h"].set_ylabel("g")
plt.colorbar(cs, ax = axs["h"])
axs["h"].set_title("correlation ratio")


cs = axs["i"].imshow(delta_rate, origin="lower", extent = (g_ii_min, g_ii_max, g_min, g_max), vmin = 1, vmax = 1.5)
axs["i"].set_xlabel("g_ii")
axs["i"].set_ylabel("g")
plt.colorbar(cs, ax = axs["i"])
axs["i"].set_title("rate ratio")




plt.tight_layout()
sns.despine(fig = fig)
plt.savefig("../results/compare_inhib_levels/plot_fixed_input.pdf")

plt.show()