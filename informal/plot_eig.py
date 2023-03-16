from src.theory import J_eff
from src.generate_connectivity import macro_weights
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def G(J, h, g, NE, NI, pE, pI):
    mw = macro_weights(J, h, h, g)
    Ns = [NE/2, NE/2, NI, NE/2, NE/2, NI]
    p_mat = np.array([[pE, pE, pI, pE, pE, pI] for i in range(6)])
    return  J_eff(mw, Ns, p_mat)

def E_submat(J):
    return J[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]

NE = 100
NI = 10
N = 2 * (NE + NI)
h = 1
pE = 0.1
pI = 0.5

J_range = np.linspace(1,40)
g_range = np.linspace(0, 3)

df = pd.DataFrame()
for J in J_range:
    for g in g_range:
        W = G(J/N, h, g, NE, NI, pE, pI)
        W_E = E_submat(W)
        lam = np.max(np.linalg.eigvals(W))
        lam_e = np.max(np.linalg.eigvals(W_E))
        df_row = pd.DataFrame({"J": [J], "g": [g], "lambda" :[lam], "lambda_E": [lam_e]})
        df = pd.concat([df, df_row], ignore_index= True)

fig, ax = plt.subplots()
sns.lineplot(data = df, x = "g", y = "lambda", hue = "J", ax = ax)
ax.hlines(1, 0,max(g_range)  )
plt.show()



