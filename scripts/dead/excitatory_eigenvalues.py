
import numpy as np
import params
from src.theory import  y_pred,  C_pred_off_diag, J_eff
import matplotlib.pyplot as plt
from src.generate_connectivity import  macro_weights
import seaborn as sb
import pandas as pd
from matplotlib import colors
import pickle as pkl


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
g = par.g
J = par.J
Ns = par.cells_per_region
y0 = par.b[0]
p_mat = par.macro_connectivity


h_vals = par.h_range

hue_order = ["Tagged vs Tagged", "Tagged vs Non-tagged", "Non-tagged vs Non-tagged"]
cor_df = pd.DataFrame(columns=["relative strength", "correlation", "pair_group"])
rate_df = pd.DataFrame(columns=["relative strength", "rate", "type"])
plt.rcParams["text.usetex"] = True
plt.rcParams['pdf.fonttype'] = 42 

with open("results/index_dict.pkl", "rb") as f:
    index_dict = pkl.load(f)

excitatory_inds = np.concatenate([index_dict["CA1E"], index_dict["CA1P"], index_dict["CA3E"], index_dict["CA3P"]])

for h in h_vals:

    with open("results/W_h={}.pkl".format(h), "rb") as f:
        W = pkl.load(f)
    
    W_E = W[np.ix_(excitatory_inds, excitatory_inds)]
    max_eig = np.max(np.linalg.eigvals(W_E))
    print(max_eig)




     
