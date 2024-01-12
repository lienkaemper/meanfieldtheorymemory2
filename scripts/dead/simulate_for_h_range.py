import numpy as np
import params_model_1
import params_model_2

from src.generate_connectivity import hippo_weights, gen_adjacency
from src.simulation import sim_glm_pop
from src.theory import y_pred_from_full_connectivity
import pickle as pkl
import gc

for i, params in enumerate([params_model_1, params_model_2]):
    i = 1
    print(i)

    print("running")
    par = params.params()
    print("g = ", par.g)
    N = par.N
    #tau = par.tau
    b = par.b
    tstop = par.tstop

    # Euler step
    cells_per_region = par.cells_per_region
    macro_connectivity = par.macro_connectivity
    g = par.g
    J = par.J
    dt = par.dt

    print(macro_connectivity)

    h_vals = par.h_range


    A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)

    with open("results/index_dict_{}.pkl".format(i), "wb") as file:
        pkl.dump(index_dict, file)

    with open("results/adjacency_{}.pkl".format(i), "wb") as file:
        pkl.dump(A, file)

    for h in h_vals:
        print("h = {}".format(h))

        W =  hippo_weights(index_dict, A, h,h, g, J)
        print(np.max(np.linalg.eigvals(W)))
        y = y_pred_from_full_connectivity(W, b, index_dict)
        print(np.max(y))
        maxspikes = int(np.max(y) * N * tstop)

        gc.collect()

        _, spktimes = sim_glm_pop(J=W, E=b, dt = dt, tstop=tstop, tstim=0, Estim=0, v_th = 0, maxspikes = maxspikes)

        print("completed simulation")

        with open("results/W_h={}i={}.pkl".format(h,i), "wb") as file:
            pkl.dump(W, file)


        with open("results/spikes_h={}i={}.pkl".format(h,i), "wb") as file:
            pkl.dump(spktimes, file)
        
        del(spktimes)
        del(_)


