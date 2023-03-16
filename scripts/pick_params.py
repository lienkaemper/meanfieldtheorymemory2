import numpy as np
import params
from src.theory import  y_pred,  C_pred_off_diag, J_eff, cor_from_full_connectivity, y_pred_from_full_connectivity
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd
import pickle as pkl
from src.generate_connectivity import hippo_weights,  gen_adjacency
from src.correlation_functions import mean_by_region



par = params.params()
N = par.N
b = par.b
tstop = par.tstop
#dt = .02 * tau  # Euler step
#p = par.p
h3_before = par.h3_before
h3_after = par.h3_after
h1_before = par.h1_before
h1_after = par.h1_after
g = 2*par.g
J = par.J/2
Ns = 5* par.cells_per_region
y0 = par.b[0]
p_mat = par.macro_connectivity

# Ns = 50*np.array([10,10,4,10,10,4])
# N = sum(Ns)
# g = 2
# J = 5/N

# y0 = 0.2

# pEE = 0.1
# pEI = 0.5
# pIE = 0.1
# pII = 0.5
# p_mat = np.array([
#              [pEE, pEE, pEI, 0, 0, 0],
#              [pEE, pEE, pEI, 0, 0, 0],
#              [pIE, pIE, pII, 0, 0, 0],
#              [pEE, pEE, 0, 0, 0, pEI],
#              [pEE, pEE, 0, 0, 0, pEI],
#              [pIE, pIE, 0, pIE, pIE, pII]])

#p_mat = np.ones((6,6))

H_range = np.linspace(1, 1.75, num = 10)

hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
cor_df = pd.DataFrame(columns=["relative strength", "correlation", "pair_group", "source"])
rate_df = pd.DataFrame(columns=["relative strength", "rate", "type"])

# with open("results/adjacency.pkl", "rb") as f:
#     A =  pkl.load(f)

A, index_dict = gen_adjacency(Ns, p_mat)



J_range = np.linspace(.25, .5)
G_range = np.linspace(2, 10)
df = pd.DataFrame(columns = ["g", "J", "C_before", "C_after"])
for J in J_range:
    for g in G_range:
        vals = np.zeros(2)
        for i, h in enumerate([1,1.5]):
            G = macro_weights(J, h, h, g)
            C = C_pred_off_diag(G, Ns, p_mat, y0)
            y = y_pred(G, Ns, p_mat, y0)
            vals[i] = C[3,3]/y[3]
        df_row = pd.DataFrame({"g": [g], "J": [J], "C_before": [vals[0]], "C_after": [vals[1]]})
        df = pd.concat([df, df_row], ignore_index=True)


df["ratio"] = df["C_after"]/df["C_before"]

fig, axs = plt.subplots(3)           
sb.lineplot(df, x = "g", y = "C_before", hue = "J", ax = axs[0])


           
sb.lineplot(df, x = "g", y = "C_after", hue= "J", ax = axs[1])

           
sb.lineplot(df, x = "g", y = "ratio", hue= "J", ax = axs[2])
plt.show()

