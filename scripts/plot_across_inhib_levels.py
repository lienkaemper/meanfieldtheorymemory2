import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import itertools


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

with open("../results/compare_inhib_levels/raw_df.pkl", "rb") as f:
    raw_df = pkl.load(f)

with open("../results/compare_inhib_levels/theory_df.pkl", "rb") as f:
    theory_df = pkl.load(f)

with open("../results/compare_inhib_levels/decomposition_df.pkl", "rb") as f:
     decomp_df = pkl.load(file = f)

with open("../results/compare_inhib_levels/index.pkl", "rb") as f:
    index_dict = pkl.load(file = f)

CA1 = list(itertools.chain(index_dict["CA1E"], index_dict["CA1P"]))
all_neurons = range(N)


# with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(3.0,1), "rb") as f:
#     spktimes_high_before = pkl.load(f)

# with open("../results/compare_inhib_levels/spktimes_g={}h={}.pkl".format(3.0,2), "rb") as f:
#     spktimes_high_after = pkl.load(f)




fig, axs = plt.subplot_mosaic([["a", "a", "b", "c"], 
                               ["d", "e", "f", "g"]], figsize = (7.2, 4))

# yticks = [r[0] for r in index_dict.values()]



# raster_plot(spktimes =spktimes_high_before, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["b"], yticks=yticks)
# #axs["b"].set_title("g = 4, h = 1")
# raster_plot(spktimes =spktimes_high_after, neurons = all_neurons, t_start  = 0, t_stop = 500, ax = axs["c"], yticks=yticks)
# #axs["c"].set_title("g = 4, h = 2")

baseline_df = raw_df.loc[raw_df["g"] ==1]
baseline_df = baseline_df.loc[:, ["h", "pred_cor_engram_vs_engram", "pred_cor_engram_vs_non_engram","pred_cor_non_engram_vs_non_engram" ]]

baseline_df = baseline_df.melt(id_vars=['h'], var_name='region', value_name='correlation')


sns.barplot(data = baseline_df, x = "region", hue = "h", y = "correlation", ax = axs["b"])
axs["b"].get_legend().remove()
axs["b"].set(xticklabels=[])
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_engram_ratio", ax = axs["c"], color = "#F37343", label = "Engram vs. engram")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_engram_ratio", ax = axs["c"], s = size, color = "#F37343")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_engram_vs_non_engram_ratio", ax = axs["c"], label = "Engram vs. non-engram", color = "#FEC20E")
sns.scatterplot(data = df, x = "g", y = "sim_cor_engram_vs_non_engram_ratio", ax = axs["c"], s = size, color = "#FEC20E")
sns.lineplot(data = theory_df, x = "g", y = "pred_cor_non_engram_vs_non_engram_ratio", ax = axs["c"], label = "Non-engram vs. non-engram", color = "#06ABC8")
sns.scatterplot(data = df, x = "g", y = "sim_cor_non_engram_vs_non_engram_ratio", ax = axs["c"], s = size, color = "#06ABC8")
axs["c"].set_xlim([1,3])
axs["c"].set_title("Correlation ratio")
axs["c"].set_xlabel("Inhibition strength: g")
axs["c"].set_ylabel("Correlation ratio")
axs["c"].get_legend().remove()



# sns.scatterplot(data = df, x = "g", y = "sim_rate_engram_ratio", ax = axs["e"], s = size, color = "#F37343")
# sns.lineplot(data = theory_df, x = "g", y = "pred_rate_engram_ratio", ax = axs["e"], label = "Engram", color = "#F37343" )
# sns.scatterplot(data = df, x = "g", y = "sim_rate_non_engram_ratio", ax = axs["e"], s = size, color = "#06ABC8")
# sns.lineplot(data = theory_df, x = "g", y = "pred_rate_non_engram_ratio", ax = axs["e"], label = "Non-engram", color = "#06ABC8" )
# axs["e"].set_title("rate ratio")
# axs["e"].set_xlabel("inhibition strength: g")
# axs["e"].set_ylabel("rate ratio")
# axs["e"].get_legend().remove()
# axs["e"].set_xlim([1,3])






J0 = .2
g_min = 1
g_max = 3
n_g = 10
gs = np.linspace(g_min, g_max, n_g)
b = np.array([.4, .4, .5, .4, .4, .5]) 
N_E =60
N_I = 15
Ns =np.array([N_E, N_E, N_I,  N_E, N_E, N_I])
p= 2
nterms = 2


# internal_before = np.array([CA1_internal_cov_offdiag(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
# inherited_before = np.array([CA1_inherited_cov(J0=J0, g=g, h=1,b=b, N=Ns, p = p)[0,0] for g in gs])
# ca3_before =  np.array([CA3_internal_cov(J0=J0, g=g, h=1,b=b, N=Ns, p =1)[0,0] for g in gs])
# total_before = internal_before  + inherited_before

# internal_after = np.array([CA1_internal_cov_offdiag(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])
# inherited_after = np.array([CA1_inherited_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])
# #inherited_after_fixed = np.array([CA1_inherited_cov_ca3_fixed(J0=J0, g=g, h=2,b=b, N=Ns, p = p) for g in gs])
# ca3_after =  np.array([CA3_internal_cov(J0=J0, g=g, h=2,b=b, N=Ns, p = p)[0,0] for g in gs])

# total_after = internal_after + inherited_after
#decomp_df = pd.DataFrame({"g" : g_list, "h" : h_list, "CA1_internal" :  CA1_internal, "CA1_inherited" : CA1_inherited, "CA3" : CA3})


sns.lineplot(data = decomp_df, x = "g", y = "CA1_internal", color = 'blue', style = "h", ax=axs["d"])

sns.lineplot(data = decomp_df, x = "g", y = "CA1_inherited", color = "green", style = "h", ax=axs["d"])
axs["d"].get_legend().remove()
axs["d"].set_xlabel("Inhibitory strength g")
axs["d"].set_ylabel("Covariance")

decomp_df["CA1_total"] = decomp_df["CA1_internal"] + decomp_df["CA1_inherited"] 
sns.lineplot(data = decomp_df, x = "g", y = "CA1_total", style = "h", color = "black", ax=axs["e"])
#axs[1].plot(gs, inherited_after_fixed, color ="black", linestyle = "--", label = "after, CA3 fixed")
axs["e"].set_title("Total")
axs["e"].legend()
axs["e"].sharey(axs["d"])
axs["e"].get_legend().remove()



decomp_df["CA3_total"] = decomp_df["from_CA3E"] + decomp_df["from_CA3N"] + decomp_df["from_CA3I"]
decomp_df["CA3_ext"] = decomp_df["from_CA3E"] + decomp_df["from_CA3N"] 
sns.lineplot(data = decomp_df, x = "g", y = "CA3_total", style="h", color= "black", ax = axs["f"])
axs["f"].set_title("Total")
axs["f"].get_legend().remove()

sns.lineplot(data = decomp_df, x = "g", y =  "CA3_ext", style = "h", ax = axs["g"] )
axs["g"].set_title("By source")
axs["g"].sharey(axs["f"])

sns.lineplot(data = decomp_df, x = "g", y =  "from_CA3I", style = "h", ax = axs["g"] )
axs["g"].sharey(axs["f"])
axs["g"].get_legend().remove()
#axs["k"].yaxis.set_ticklabels([])



# fig.supxlabel("Inhibitory strength g")
# fig.suptitle("CA3 engram-engram covariance")
sns.despine(fig = fig)
plt.tight_layout(w_pad = .001)
plt.savefig("../results/CA1_cov_sources.pdf")
plt.show()