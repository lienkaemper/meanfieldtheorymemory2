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

with open("../results/compare_inhib_levels/theory_df.pkl", "rb") as f:
    theory_df = pkl.load(f)


with open("../results/compare_inhib_levels/index.pkl", "rb") as f:
    index_dict = pkl.load(file = f)

CA1 = list(itertools.chain(index_dict["CA1E"], index_dict["CA1P"]))
all_neurons = range(N)


with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(4.0,1), "rb") as f:
    spktimes_high_before = pkl.load(f)

with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(4.0,2), "rb") as f:
    spktimes_high_after = pkl.load(f)




fig, axs = plt.subplot_mosaic([["b", "b", "c", "c"], 
                               ["d", "e", "f", "g"], 
                               ["h", "i", "j", "k"]], figsize = (6, 6), height_ratios = [1.5, 1, 1])

yticks = [r[0] for r in index_dict.values()]



raster_plot(spktimes =spktimes_high_before, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["b"], yticks=yticks)
#axs["b"].set_title("g = 4, h = 1")
raster_plot(spktimes =spktimes_high_after, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["c"], yticks=yticks)
#axs["c"].set_title("g = 4, h = 2")


sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_engram_ratio", ax = axs["d"], label = "Engram vs. engram")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_engram_ratio", ax = axs["d"])
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_non_engram_ratio", ax = axs["d"], label = "Engram vs. non-engram")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_non_engram_ratio", ax = axs["d"])
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_non_engram_vs_non_engram_ratio", ax = axs["d"], label = "Non-ngram vs. non-engram")
sns.scatterplot(data = df, x = "g", y = "sim_cor_non_engram_vs_non_engram_ratio", ax = axs["d"])
axs["d"].set_title("Correlation ratio")
axs["d"].set_xlabel("Inhibition strength: g")
axs["d"].set_ylabel("Correlation ratio")
axs["d"].legend("off")
axs["d"].get_legend().remove()



sns.scatterplot(data = df, x = "g", y = "sim_rate_engram_ratio", ax = axs["e"])
sns.lineplot(data = theory_df, x = "g", y = "pred_rate_engram_ratio", ax = axs["e"], label = "Engram" )
sns.scatterplot(data = df, x = "g", y = "sim_rate_non_engram_ratio", ax = axs["e"])
sns.lineplot(data = theory_df, x = "g", y = "pred_rate_non_engram_ratio", ax = axs["e"], label = "Non-engram" )
axs["e"].set_title("rate ratio")
axs["e"].set_xlabel("inhibition strength: g")
axs["e"].set_ylabel("rate ratio")
axs["e"].get_legend().remove()





sns.despine(fig = fig)
plt.tight_layout(w_pad = .01)

plt.savefig("../results/compare_inhib_levels/plot_fixed_input.pdf")

plt.show()