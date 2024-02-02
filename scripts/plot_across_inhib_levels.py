import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import itertools

from src.simulation import sim_glm_pop
from src.theory import y_pred_full, covariance_full,  y_0_quad,  find_iso_rate, y_corrected_quad, find_iso_rate_input, cor_pred, CA1_internal_cov, CA1_internal_cov_offdiag,  CA3_internal_cov, CA1_inherited_cov,  CA3_E_from_E, CA3_E_from_N, CA3_E_from_I
from src.correlation_functions import rate, mean_by_region, tot_cross_covariance_matrix, two_pop_correlation, mean_pop_correlation, cov_to_cor
from src.plotting import raster_plot, abline
from src.generate_connectivity import excitatory_only, gen_adjacency, hippo_weights, macro_weights
from src.plotting import raster_plot
plt.style.use('paper_style.mplstyle')
size = 20


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


sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_engram_h=2", ax = axs["d"], color = "#F37343", label = "Engram vs. engram")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_engram_h=2", ax = axs["d"], s = size, color = "#F37343")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_non_engram_h=2", ax = axs["d"], label = "Engram vs. non-engram", color = "#FEC20E")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_non_engram_h=2", ax = axs["d"], s = size, color = "#FEC20E")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_non_engram_vs_non_engram_h=2", ax = axs["d"], label = "Non-engram vs. non-engram", color = "#06ABC8")
sns.scatterplot(data = df, x = "g", y = "sim_cor_non_engram_vs_non_engram_h=2", ax = axs["d"], s = size, color = "#06ABC8")


sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_engram_h=1", ax = axs["d"], color = "#F37343", label = "Engram vs. engram", linestyle = "--")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_engram_h=1", ax = axs["d"], s = size, color = "#F37343", marker = "*")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_non_engram_h=1", ax = axs["d"], label = "Engram vs. non-engram", color = "#FEC20E", linestyle = "--")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_non_engram_h=1", ax = axs["d"], s = size, color = "#FEC20E", marker = "*")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_non_engram_vs_non_engram_h=1", ax = axs["d"], label = "Non-engram vs. non-engram", color = "#06ABC8", linestyle = "--")
sns.scatterplot(data = df, x = "g", y = "sim_cor_non_engram_vs_non_engram_h=1", ax = axs["d"], s = size, color = "#06ABC8",marker = "*")
axs["d"].set_title("Correlation ratio")
axs["d"].set_xlabel("Inhibition strength: g")
axs["d"].set_ylabel("Correlation ratio")
axs["d"].legend("off")
axs["d"].get_legend().remove()



sns.scatterplot(data = df, x = "g", y = "sim_rate_engram_ratio", ax = axs["e"], s = size, color = "#F37343")
sns.lineplot(data = theory_df, x = "g", y = "pred_rate_engram_ratio", ax = axs["e"], label = "Engram", color = "#F37343" )
sns.scatterplot(data = df, x = "g", y = "sim_rate_non_engram_ratio", ax = axs["e"], s = size, color = "#06ABC8")
sns.lineplot(data = theory_df, x = "g", y = "pred_rate_non_engram_ratio", ax = axs["e"], label = "Non-engram", color = "#06ABC8" )
axs["e"].set_title("rate ratio")
axs["e"].set_xlabel("inhibition strength: g")
axs["e"].set_ylabel("rate ratio")
axs["e"].get_legend().remove()







J0 = .2
g_min = 1
g_max = 4
n_g = 10
gs = np.linspace(g_min, g_max, n_g)
b = np.array([.5, .5, .7, .5, .5, .7])
N_E =60
N_I = 15
Ns =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
p= 2
nterms = 2


internal_before = np.array([CA1_internal_cov_offdiag(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
inherited_before = np.array([CA1_inherited_cov(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
ca3_before =  np.array([CA3_internal_cov(J0=J0, g=g, h=1,b=b, N=Ns, p =1)[0,0] for g in gs])
total_before = internal_before  + inherited_before

internal_after = np.array([CA1_internal_cov_offdiag(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])
inherited_after = np.array([CA1_inherited_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])
#inherited_after_fixed = np.array([CA1_inherited_cov_ca3_fixed(J0=J0, g=g, h=2,b=b, N=Ns, p = p) for g in gs])

ca3_after =  np.array([CA3_internal_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])

total_after = internal_after + inherited_after


axs["f"].plot(gs, internal_before, color ="gray", label = "before")
axs["f"].plot(gs, internal_after, color ="black", label = "after")
#axs['f'].legend()
axs["f"].set_title("Internally generated")
axs["f"].set_xlabel("Inhibitory strength g")
axs["f"].set_ylabel("Covariance")

axs["f"].plot(gs, inherited_before, color ="gray", label = "before")
axs["f"].plot(gs, inherited_after, color ="black", label = "after")
#axs[1].plot(gs, inherited_after_fixed, color ="black", linestyle = "--", label = "after, CA3 fixed")
axs["f"].set_title("From CA3")
#axs["f"].legend()

axs["g"].plot(gs, inherited_before + internal_before , color ="gray", label = "before")
axs["g"].plot(gs, inherited_after + internal_after, color ="black", label = "after")
#axs[1].plot(gs, inherited_after_fixed, color ="black", linestyle = "--", label = "after, CA3 fixed")
axs["g"].set_title("Total")
axs["g"].legend()
axs["g"].sharey(axs["f"])



# fig, ax = plt.subplots(figsize = (3,3))
# axs["k"].plot(gs, ca3_before, color ="gray", label = "before")
# ax.plot(gs, ca3_after, color ="black", label = "after")
# ax.legend()
# fig.suptitle("CA3, Engram-Engram covariance")
# fig.supxlabel("Inhibitory strength g")
# fig.supylabel("Covariance")
# plt.tight_layout()  
# plt.show()



I_before = np.array([CA3_E_from_I(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p) for g in gs])
I_after = np.array([CA3_E_from_I(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p) for g in gs])
I_before_approx = np.array([CA3_E_from_I(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p, nterms = nterms) for g in gs])
I_after_approx = np.array([CA3_E_from_I(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p, nterms = nterms) for g in gs])

E_before= np.array([CA3_E_from_E(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p) for g in gs])
E_after = np.array([CA3_E_from_E(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p) for g in gs])
E_before_approx= np.array([CA3_E_from_E(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p, nterms = nterms) for g in gs])
E_after_approx= np.array([CA3_E_from_E(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p, nterms = nterms) for g in gs])

N_before =  np.array([CA3_E_from_N(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p) for g in gs])
N_after = np.array([CA3_E_from_N(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p) for g in gs])
N_before_approx =  np.array([CA3_E_from_N(J0 = .2, g = g, h = 1, b=b, N=Ns, p = p, nterms = nterms) for g in gs])
N_after_approx = np.array([CA3_E_from_N(J0 = .2, g = g, h = 2, b=b, N=Ns, p = p, nterms = nterms) for g in gs])

tot_before=  np.array([CA3_internal_cov(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
tot_after =  np.array([CA3_internal_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])

axs["h"].set_ylabel("covariance")
axs["h"].plot(gs,E_before + N_before + I_before, label = "before", color = "gray")
axs["h"].plot(gs, E_after + N_after + I_after, label = "after", color = "black")
#axs["h"].plot(gs,E_before_approx + N_before_approx + I_before_approx, label = "before", color = "gray", linestyle ="--")
#axs["h"].plot(gs, E_after_approx + N_after_approx + I_after_approx, label = "after", color = "black", linestyle = "--")
axs["h"].set_title("Total")


axs["i"].plot(gs, E_before, label = "before",color =  "#F37343", alpha = .5)
axs["i"].plot(gs, E_after, label = "after",color =  "#F37343", alpha = 1)
# axs["i"].plot(gs, E_before_approx, label = "before",color = "gray", linestyle = "--")
# axs["i"].plot(gs, E_after_approx, label = "after", color = "black", linestyle = "--")
axs["i"].set_title("Engram")
axs["i"].sharey(axs["h"])


axs["j"].plot(gs, N_before, label = "before", color = "#06ABC8", alpha = .5)
axs["j"].plot(gs, N_after, label = "after",  color = "#06ABC8", alpha = 1)
# axs["j"].plot(gs, N_before_approx, label = "before", color = "gray", linestyle = "--")
# axs["j"].plot(gs, N_after_approx, label = "after", color = "black", linestyle = "--")
axs["j"].set_title("Non-engram")
axs["j"].sharey(axs["h"])

axs["k"].plot(gs, I_before, label = "before", color = "gray")
axs["k"].plot(gs, I_after, label = "after", color = "black")
# axs["k"].plot(gs, I_before_approx, label = "before", color = "gray", linestyle = "--")
# axs["k"].plot(gs, I_after_approx, label = "after", color = "black", linestyle = "--")
axs["k"].set_title("Inhibitory")
axs["k"].sharey(axs["h"])
#axs["k"].yaxis.set_ticklabels([])



# fig.supxlabel("Inhibitory strength g")
# fig.suptitle("CA3 engram-engram covariance")
sns.despine(fig = fig)
plt.tight_layout(w_pad = .001)
plt.savefig("../results/CA1_cov_sources.pdf")
plt.show()