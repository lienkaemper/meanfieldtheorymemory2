import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

N = 10
M = 20
T = 50
p = 0.2

fig, ax = plt.subplots()

ds = []
preds = []
sims = []

for i in range(T):
    A = np.random.rand(N, N) < p
    A = A.astype(int)
    B = np.eye(N)
    Bs = [B]
    for m in range(M):
        B = B @ A
        Bs.append(B)

    for m, B in enumerate(Bs):
        pred_val = N**(m-1) * p**(m) 
        vals = Bs[m].flatten()
        ds.extend(m*np.ones_like(vals))
        preds.extend(pred_val*np.ones_like(vals))
        sims.extend(vals)
        

df = pd.DataFrame({"d": ds, "pred": preds, "sims":sims})
df_noise = df.copy()
print(max(df["d"]))

sns.scatterplot(x = "pred", y = "sims", data = df, hue= 'd', alpha = 0.1 )
sns.scatterplot(x = "pred", y = "sims", data = df.groupby("d").mean(), hue= 'd', alpha = 1 )
plt.axline((0,0), slope = 1.0, color = 'black')
plt.show()


sns.lineplot(x = "pred", y = "sims", data = df, hue= 'd', alpha = 1 )
plt.axline((0,0), slope = 1.0, color = 'black')
plt.show()