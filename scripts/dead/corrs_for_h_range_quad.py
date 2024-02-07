import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import gc
import os
import sys
from tqdm import tqdm 
from itertools import chain 


from src.correlation_functions import  tot_cross_covariance



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




cov_l = []
cov_q = []
cor_l = []
cor_q = []
h_list = []
region_i_list = []
region_j_list = []

for h in h_range:
    
    with open(dirname + "/spikes_q_h={}.pkl".format(h), "rb") as file:
        spktimes_q = pkl.load(file)

    with open(dirname + "/spikes_l_h={}.pkl".format(h), "rb") as file:
        spktimes_l = pkl.load( file)


    inds = [ind for ind in chain(index_dict["CA1E"], index_dict["CA1P"])]
    n_E = len(index_dict["CA1E"])
    n_P = len(index_dict["CA1P"])
    n = n_E + n_P
    cov_mat_l = np.zeros((n,n))
    cov_mat_q = np.zeros((n, n))
    for mat_i, i in tqdm(enumerate(inds)):
        for mat_j, j in enumerate(inds):
            if mat_i <= mat_j:
                val_l = tot_cross_covariance(spktimes_l, i, j, dt, tstop)
                val_q =  tot_cross_covariance(spktimes_q, i, j, dt, tstop)
                cov_mat_l[mat_i, mat_j] = val_l
                cov_mat_q[mat_i, mat_j] = val_q 
                cov_mat_l[mat_j, mat_i] = val_l
                cov_mat_q[mat_j, mat_i] = val_q

    cor_mat_l = (1/np.sqrt(np.diag(cov_mat_l))) *cov_mat_l * (1/(np.sqrt(np.diag(cov_mat_l))))[...,None]
    cor_mat_q = (1/np.sqrt(np.diag(cov_mat_q))) *cov_mat_q * (1/(np.sqrt(np.diag(cov_mat_q))))[...,None]


    for mat_i, i in enumerate(inds):
        for mat_j,j in enumerate(inds):
            if i< j:
                cov_l.append(cov_mat_l[mat_i, mat_j])
                cov_q.append(cov_mat_q[mat_i, mat_j])
                cor_l.append(cor_mat_l[mat_i, mat_j])
                cor_q.append(cor_mat_q[mat_i, mat_j])
                h_list.append(h)
                if i in index_dict["CA1E"]:
                    region_i_list.append("CA1E")
                if j in index_dict["CA1E"]:
                    region_j_list.append("CA1E")
                if i in index_dict["CA1P"]:
                    region_i_list.append("CA1P")
                if j in index_dict["CA1P"]:
                    region_j_list.append("CA1P")




df = pd.DataFrame({"region_i": region_i_list, "region_j": region_j_list, "h": h_list, "cov_q": cov_q, "cov_l": cov_l,  "cor_q": cor_q, "cor_l": cor_l})
df.to_csv(dirname+"/lin_vs_quad_cor.csv")
