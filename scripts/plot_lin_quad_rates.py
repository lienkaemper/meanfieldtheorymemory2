import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dir = f.read()
else:
    dir = sys.argv[1]

fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True)
df = pd.read_csv(dir + "/lin_vs_quad.csv")

df.pred_rate_q = [complex(df.pred_rate_q[i]).real for i in range(len(df.pred_rate_q))]
df.input = [complex(df.input[i]).real for i in range(len(df.input))]


print(df.dtypes)
CA1_df = df[df.region.isin(["CA1E", "CA1P"])]
print(df.head)

axs[0].set_title("linear")
sns.lineplot(data = CA1_df ,  x = "h", y = "rate_l", hue = "region", ax = axs[0] )
sns.lineplot(data = CA1_df ,  x = "h", y = "pred_rate_l", hue = "region", ax = axs[0], linestyle='--' )


axs[1].set_title("quadratic")
sns.lineplot(data = CA1_df , x = "h", y = "rate_q", hue = "region" , ax = axs[1])
sns.lineplot(data = CA1_df , x = "h", y = "pred_rate_q", hue = "region" , ax = axs[1], linestyle='--')

plt.savefig(dir + "/rates_lin_and_quad.pdf")
plt.show()



sns.histplot(data = CA1_df, x = "input")
plt.savefig(dir + "/hist_lin_and_quad.pdf")

plt.show()

