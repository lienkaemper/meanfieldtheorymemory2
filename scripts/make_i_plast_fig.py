import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl



plt.style.use('paper_style.mplstyle')
size = 20

with open("../results/compare_inhib_levels/plast_df.pkl", "rb") as f:
    plast_df = pkl.load( file = f)

with open("../results/compare_inhib_levels/raw_df.pkl", "rb") as f:
    raw_df = pkl.load(f)

    
plast_df= plast_df[plast_df["g"] == 1]


baseline_df = raw_df.loc[raw_df["g"] ==1]
baseline_df = baseline_df.loc[:, ["h", "sim_rate_engram" , "sim_rate_non_engram"  ]]

baseline_df = baseline_df.melt(id_vars=['h'], var_name='region', value_name='rate')
baseline_rate = np.mean(baseline_df.loc[(baseline_df['h'] ==1) & ( baseline_df["region"] == "sim_rate_engram"), "rate"])
print(baseline_rate)
baseline_df["norm_rate"] =(baseline_df["rate"]/baseline_rate)


fig, axs = plt.subplots(1,3, figsize = (7,2))
sns.lineplot(data = plast_df, x = "h", y = "h_i3", ax = axs[1], label = "CA3")
sns.lineplot(data = plast_df, x = "h", y = "h_i1", ax = axs[1], label = "CA1")

sns.barplot(data = baseline_df, x ="region", hue = "h", y = "norm_rate", ax = axs[2])
plt.tight_layout()
plt.savefig("../results/inhib_plast_fig.pdf")
plt.show()

