import numpy as np
import params
from src.generate_connectivity import hippo_weights
from src.simulation import sim_glm_pop
import pickle as pkl

print("run 1: before learning")
par = params.params()
N = par.N
tau = par.tau
b = par.b
gain = par.gain
trans = par.trans
tstop = par.tstop

dt = .02 * tau  # Euler step


cells_per_region = par.cells_per_region
macro_connectivity = par.macro_connectivity
h1 = par.h1_before
h3 = par.h3_before
g = par.g
J = par.J
np.random.seed(10)
W, index_dict = hippo_weights(cells_per_region,  macro_connectivity, h3, h1, g, J)

print("made adjancency matrix ")


_, spktimes = sim_glm_pop(J=W, E=b, tstop=tstop, tstim=0, Estim=0, v_th = 0)

print("completed simulation")

file = open("results/before_learning_W.pkl", "wb")
pkl.dump(W, file)
file.close()

file = open("results/index_dict_before.pkl", "wb")
pkl.dump(index_dict, file)
file.close()


file = open("results/before_learning_spikes.pkl", "wb")
pkl.dump(spktimes, file)
file.close()




print("run 2: after learning")



h1 = par.h1_after
h3 = par.h3_after
np.random.seed(10)

W, index_dict = hippo_weights(cells_per_region,  macro_connectivity, h3, h1, g, J)
print("made adjancency matrix ")



_, spktimes = sim_glm_pop(J=W, E=b, tstop=tstop, tstim=0, Estim=0, v_th = 0)

print("completed simulation")

file = open("results/after_learning_W.pkl", "wb")
pkl.dump(W, file)
file.close()

file = open("results/index_dict_after.pkl", "wb")
pkl.dump(index_dict, file)
file.close()

file = open("results/after_learning_spikes.pkl", "wb")
pkl.dump(spktimes, file)
file.close()
