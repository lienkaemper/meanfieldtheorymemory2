import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('paper_style.mplstyle')


rate_df = pd.read_csv("rate_df_low_inhib.csv")
cor_df = pd.read_csv("cor_df_low_inhib.csv")
pred_rate_df = pd.read_csv("pred_rates.csv")
cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]


pred_rate_df = pred_rate_df[pred_rate_df["region"].isin(["CA1E", "CA1P"])]
baseline_rate = np.mean(pred_rate_df[pred_rate_df["h"] == 1]["pred_rate"])

norm_pred_rate_df = pred_rate_df.copy()
norm_pred_rate_df["pred_rate"] = norm_pred_rate_df["pred_rate"]/baseline_rate
norm_rate_df = rate_df.copy()
norm_rate_df["rate"] = rate_df["rate"]/baseline_rate

fig, axs = plt.subplots(1,2, figsize = (6,2))
sns.barplot(data= norm_rate_df[norm_rate_df["h"].isin([1.0,2.0])], x = "region", hue = "h", y = "rate", ax = axs[0], palette = ["gray", "black"])
axs[0].get_legend().remove()

sns.barplot(data= cor_df[cor_df["h"].isin([1.0,2.0])], x = "regions", hue = "h", y = "correlation", ax = axs[1],palette = ["gray", "black"])
axs[1].get_legend().remove()
plt.tight_layout()
plt.show()

def raster_plot(spktimes, neurons, t_start, t_stop, yticks = None, ax = None):
    df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
    df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
    df = df[df["neuron"].isin(neurons)]
    if ax is None:
        fig, ax = plt.subplots()
    s = 1000
    sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/(2*len(neurons)), ax = ax, hue = "neuron",  palette = ["black"])
    plt.legend([],[], frameon=False)
    if yticks is not None:
        ax.set_yticks(yticks)    
    if ax is None:
        return fig, ax