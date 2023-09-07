import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('poster_style.mplstyle')


corr_cfc = pd.read_csv("../data/corr_mean_animal_cfc.csv")
corr_tfc = pd.read_csv("../data/corr_mean_animal_tfc.csv")
corr_hc = pd.read_csv("../data/corr_mean_animal_hc.csv")

rate_cfc = pd.read_csv("../data/rate_mean_animal_cfc.csv")
rate_tfc = pd.read_csv("../data/rate_mean_animal_tfc.csv")
rate_hc = pd.read_csv("../data/rate_mean_animal_hc.csv")


def reshape_df(df, mice_2_cols, new_names, value_name):
    dfs = []
    for mouse in mice_2_cols.keys():
        mouse_df = df[["condition"] + mice_2_cols[mouse]].copy()
        mouse_df.columns =  new_names
        mouse_df["mouse"] = mouse
        dfs.append(mouse_df)
    df = pd.concat(dfs)
    df = pd.melt(df, id_vars=['condition', 'mouse'], value_vars=new_names[1:], var_name='region', value_name=value_name)
    return df


mice_2_cols = {0: ["EE", "NN", "EN"], 1: ["EE.1", "NN.1", "EN.1"], 2: ["EE.2", "NN.2", "EN.2"]}
new_names = [ "condition", "EE", "NN", "EN"]
for df in [corr_cfc,corr_tfc, corr_hc]:
    df.columns.values[0] = "condition"

corr_cfc = reshape_df(corr_cfc, mice_2_cols, new_names, "correlation")
corr_tfc = reshape_df(corr_tfc, mice_2_cols, new_names, "correlation")
corr_hc = reshape_df(corr_hc, mice_2_cols, new_names, "correlation")


mice_2_cols = {0: ["Tagged", "Non-tagged"], 1: ["Tagged.1", "Non-tagged.1"], 2: ["Tagged.2", "Non-tagged.2"]}
new_names = [ "condition", "Tagged", "Non-tagged"]
for df in [rate_cfc,rate_tfc, rate_hc]:
    df.columns.values[0] = "condition"

rate_cfc = reshape_df(rate_cfc, mice_2_cols, new_names, "rate")
rate_tfc = reshape_df(rate_tfc, mice_2_cols, new_names, "rate")
rate_hc = reshape_df(rate_hc, mice_2_cols, new_names, "rate")


plt.figure(figsize=(6,7)) 
sns.barplot(data = corr_cfc, x = "region", hue = "condition", y = "correlation")
plt.title("CFC")
plt.savefig("../results/mouse_correlation_cfc.pdf")

plt.figure(figsize=(6,7)) 
sns.barplot(data = corr_tfc, x = "region", hue = "condition", y = "correlation")
plt.title("TFC")
plt.savefig("../results/mouse_correlation_tfc.pdf")

plt.figure(figsize=(6,7)) 
sns.barplot(data = corr_hc, x = "region", hue = "condition", y = "correlation")
plt.title("HC")
plt.savefig("../results/mouse_correlation_hc.pdf")


plt.figure(figsize=(6,7)) 
sns.barplot(data = rate_cfc, x = "region", hue = "condition", y = "rate")
plt.title("CFC")
plt.savefig("../results/mouse_rate_cfc.pdf")

plt.figure(figsize=(6,7)) 
sns.barplot(data = rate_tfc, x = "rate", hue = "condition", y = "rate")
plt.savefig("../results/mouse_rate_tfc.pdf")

plt.figure(figsize=(6,7)) 
plt.title("TFC")
sns.barplot(data = rate_hc, x = "region", hue = "condition", y = "rate")
plt.title("HC")
plt.savefig("../results/mouse_rate_hc.pdf")
