
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
plt.style.use('poster_style.mplstyle')






rate_df = pd.read_csv("../results/excitatory_only_rates.csv")
cor_df = pd.read_csv("../results/excitatory_only_corrs.csv")

rate_df['region'] = rate_df['i'].apply(lambda x: 'engram' if x <= 75 else 'non-engram')
cor_df['region_i'] = cor_df['i'].apply(lambda x: 'engram' if x <= 75 else 'non-engram')
cor_df['region_j'] = cor_df['j'].apply(lambda x: 'engram' if x <= 75 else 'non-engram')


cor_df["regions"] = cor_df["region_i"] +"\n"+ cor_df["region_j"]
cor_df = cor_df[cor_df[ "i"]!= cor_df["j"]]
sns.barplot(data= rate_df, x = "region", hue = "h", y = "rate")
plt.savefig("../results/rates_ext_barplot.pdf")
plt.show()

sns.barplot(data= cor_df, x ="regions", hue = "h", y = "correlation")
plt.savefig("../results/cor_ext_barplot.pdf")

plt.show()

sns.scatterplot(data= cor_df, x = "correlation_pred", y = "correlation")
plt.show()

sns.scatterplot(data= cor_df, x = "covariance_pred", y = "covariance")
plt.show()

sns.scatterplot(data= rate_df, x = "rate_pred", y = "rate")
plt.show()