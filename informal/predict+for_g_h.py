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
g = 10*par.g
J = par.J/2
Ns = 3*par.cells_per_region
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


W =  hippo_weights(index_dict, A, 1,1, g, J)
excitatory_inds = np.concatenate([index_dict["CA1E"], index_dict["CA1P"], index_dict["CA3E"], index_dict["CA3P"]])  
W_E = W[np.ix_(excitatory_inds, excitatory_inds)]
max_eig = np.max(np.linalg.eigvals(W_E))
print("without inhibition", max_eig)
print("with inhibition", np.max(np.linalg.eigvals(W)))
J_range = [.25, .25001, .2500001]
G_range = [1,5]
for J in J_range:
    A, index_dict = gen_adjacency(Ns, p_mat)
    for g in G_range:
        for h in H_range:

            G = macro_weights(J, h, h, g)
            C = C_pred_off_diag(G, Ns, p_mat, y0)
            y = y_pred(G, Ns, p_mat, y0)
            W =  hippo_weights(index_dict, A, h,h, g, J)
            C_full = cor_from_full_connectivity(W, y0, index_dict)
            C_full =  mean_by_region(C_full, index_dict)
            y_full =y_pred_from_full_connectivity(W, y0, index_dict)
            y_full = mean_by_region(y_full, index_dict)
            cor_df_rows = pd.DataFrame({"relative strength":[h, h, h, h, h, h], 
                            "covariance": [C[3,3],C[3,4], C[4,4], C[0,0], C[0,1], C[1,1]],
                            "correlation": [C[3,3]/y[3],C[3,4]/(np.sqrt(y[3]*y[4])), C[4,4]/y[4], C[0,0]/y[0],C[0,1]/(np.sqrt(y[0]*y[1])), C[1,1]/y[1]],
                            "pair_group": ["Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged", "Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"],
                            "region": ["CA1", "CA1", "CA1", "CA3", "CA3", "CA3"], 
                            "source": ["reduced", "reduced", "reduced", "reduced", "reduced", "reduced"],
                            "J": [J, J, J, J, J, J],
                            "g": [g,g,g,g,g,g] })
            cor_df_rows_full = pd.DataFrame({"relative strength":[h, h, h, h, h, h], 
                            "covariance": [C_full[3,3],C_full[3,4], C_full[4,4], C_full[0,0], C_full[0,1], C_full[1,1]],
                            "correlation": [C_full[3,3]/y[3],C_full[3,4]/(np.sqrt(y[3]*y[4])), C_full[4,4]/y[4], C_full[0,0]/y[0],C[0,1]/(np.sqrt(y[0]*y[1])), C[1,1]/y[1]],
                            "pair_group": ["Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged", "Tagged vs Tagged",  "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"],
                            "region": ["CA1", "CA1", "CA1", "CA3", "CA3", "CA3"], 
                            "source": ["full", "full", "full", "full", "full", "full"],
                            "J": [J, J, J, J, J, J],
                            "g": [g,g,g,g,g,g] })
            cor_df = pd.concat([cor_df, cor_df_rows, cor_df_rows_full], ignore_index = True)
            rate_df_rows = pd.DataFrame({"relative strength":[h, h, h, h], 
                            "rate":[y[3], y[4], y[0], y[1] ],
                            "type": ["tagged", "non-tagged", "tagged", "non-tagged"],
                            "region": ["CA1", "CA1", "CA3", "CA3"],
                            "source": ["reduced", "reduced", "reduced", "reduced"],
                            "J": [J, J, J, J],
                            "g": [g,g,g,g] })
            rate_df_rows_full = pd.DataFrame({"relative strength":[h, h, h, h], 
                            "rate":[y_full[3], y_full[4], y_full[0], y_full[1] ],
                            "type": ["tagged", "non-tagged", "tagged", "non-tagged"],
                            "region": ["CA1", "CA1", "CA3", "CA3"],
                            "source": ["full", "full", "full", "full"],
                            "J": [J, J, J, J],
                            "g": [g,g,g,g] })
            rate_df = pd.concat([rate_df, rate_df_rows, rate_df_rows_full], ignore_index= True)

cor_df = cor_df[cor_df["region"] == "CA1"]
g = sb.FacetGrid(cor_df,  col= "g", row = "J")
g.map_dataframe(sb.lineplot, x = "relative strength", y = "correlation", hue ="pair_group", style = "source",  palette='Set2')
g.add_legend()
plt.savefig("results/cor_for_g_big.pdf")      
plt.show()


rate_df = rate_df[rate_df["region"] == "CA1"]
g = sb.FacetGrid(rate_df, col= "g", row = "J")
g.map_dataframe(sb.lineplot, x = "relative strength", y = "rate", hue = 'type', style = "source", palette='Set2')
g.add_legend()
plt.savefig("results/rate_for_g_big.pdf")
plt.show()