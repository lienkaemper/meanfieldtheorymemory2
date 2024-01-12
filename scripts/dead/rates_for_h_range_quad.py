import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import sys


from src.theory import y_pred_from_full_connectivity, y_corrected_quad, y_0_quad
from src.correlation_functions import rate, mean_by_region



if len(sys.argv) < 2:
    f = open("../results/most_recent.txt", "r")
    dirname = f.read()
else:
    dirname = sys.argv[1]


with open(dirname+"/index_dict.pkl", "rb") as file:
    index_dict = pkl.load(file)

with open(dirname + "/param_dict.pkl", "rb") as file:
    param_dict = pkl.load(file)

b = param_dict["b"]
N = param_dict["N"]
h_min = param_dict["h_min"]
h_max = param_dict["h_max"]
n_h = param_dict["n_h"]
dt = param_dict["dt"]
tstop = param_dict["tstop"]

h_range = np.linspace(h_min, h_max, n_h)


region_list = ['' for i in range(N)]
inputs_quad = []
for key in index_dict:
    start = min(index_dict[key])
    end = max(index_dict[key])
    region_list[start:end+1] = (end+1-start)*[key]


region_list = len(h_range) * region_list

h_list = [h for h in h_range for i in range(N)]




rates_l = []
rates_q = []
for h in h_range:

    with open(dirname + "/spikes_q_h={}.pkl".format(h), "rb") as file:
        spktimes_q = pkl.load(file)

    with open(dirname + "/spikes_l_h={}.pkl".format(h), "rb") as file:
        spktimes_l = pkl.load( file)


    rates_q.extend([rate(spktimes_q, i,  tstop = tstop) for i in range(N)])
    rates_l.extend([rate(spktimes_l, i, tstop = tstop) for i in range(N)])

    

df = pd.DataFrame({"region": region_list, "h": h_list, "rate_l": rates_l, "rate_q": rates_q})
df.to_csv(dirname+"/lin_vs_quad.csv")
