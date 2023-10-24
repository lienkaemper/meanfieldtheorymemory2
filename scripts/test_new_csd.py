import numpy as np

from src.correlation_functions import create_spike_train_matrix, create_spike_train
from src.simulation import sim_glm_pop
import matplotlib.pyplot as plt 
from scipy import signal
from src.plotting import raster_plot
import time 



def tot_cross_covariance(spktimes, i, j, dt, tstop ):
    
    spk_i = create_spike_train(spktimes, neuron=i, dt=dt, tstop=tstop)
    spk_j = create_spike_train(spktimes, neuron=j, dt=dt, tstop=tstop)

    spk_i -= np.mean(spk_i)
    spk_j -= np.mean(spk_j)
    _, Ctmp = signal.csd(spk_i, spk_j, fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False)

    return Ctmp[0]


def tot_cross_covariance_matrix(spktimes, inds, dt, tstop):
    n = len(inds)
    C = np.zeros((n,n))
    for c_i, i in enumerate(inds):
        for c_j, j in enumerate(inds):
            if i <= j:
                C[c_i,c_j] = np.real(tot_cross_covariance(spktimes,i,j,dt,tstop))
                C[c_j,c_i] = C[c_i, c_j]
    return C



def create_pop_spike_train(spktimes, neurons, dt=.01, tstop=100):
    Nt = int(tstop/dt)+1
    spktrain = np.zeros((Nt,))
    for neuron in neurons:
        spktimes_tmp = spktimes[spktimes[:, 1] == neuron][:, 0]
        spk_indices = spktimes_tmp / dt
        spk_indices = spk_indices.astype('int')
        spktrain[spk_indices] += 1/dt
    return spktrain

def new_create_pop_spike_train(spktimes, neurons, dt=.01, tstop=100, scaling = None):
    Nt = int(tstop/dt)+1
    spktrain = np.zeros((Nt,))
    if scaling == None:
        scaling = np.ones(len(neurons))
    for i, neuron in enumerate(neurons):
        spktimes_tmp = spktimes[spktimes[:, 1] == neuron][:, 0]
        spk_indices = spktimes_tmp / dt
        spk_indices = spk_indices.astype('int')
        spktrain[spk_indices] += scaling[i]/dt
    return spktrain

def new_pop_correlation(spktimes, neurons, dt, tstop):
    N = len(neurons)
    spiketrain = create_spike_train_matrix(spktimes, neurons, dt, tstop)
    spiketrain = spiketrain-np.mean(spiketrain, axis=1, keepdims=True)
    _, Ctmp = signal.csd(spiketrain, spiketrain, fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False, axis = 1)
    vars = Ctmp[:,0]
    vars[vars== 0] = 1
    scaling = 1/(np.sqrt(vars))
    spiketrain = spiketrain * scaling[...,None]
    spiketrain = np.sum(spiketrain, axis = 0)
    _, Ctmp = signal.csd(spiketrain, spiketrain, fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False)
    return np.real((Ctmp[0]-N)/(N*(N-1)))


def two_pop_correlation(spktimes, neurons1, neurons2, dt, tstop):
    N1 = len(neurons1)
    N2 = len(neurons2)
    pop_spiketrains = []
    for pop in [neurons1, neurons2]:
        print(neurons1)
        spiketrain = create_spike_train_matrix(spktimes,pop, dt, tstop)
        spiketrain = spiketrain-np.mean(spiketrain, axis=1, keepdims=True)
        _, Ctmp = signal.csd(spiketrain, spiketrain, fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False, axis = 1)
        vars = Ctmp[:,0]
        vars[vars== 0] = 1
        scaling = 1/(np.sqrt(vars))
        spiketrain = spiketrain * scaling[...,None]
        spiketrain = np.sum(spiketrain, axis = 0)
        pop_spiketrains.append(spiketrain)
    _, Ctmp = signal.csd(pop_spiketrains[0], pop_spiketrains[1], fs=1/dt, scaling='density', window='bartlett', nperseg=2048, return_onesided=False, detrend=False)
    return np.real(Ctmp[0]/(N1*N2))


def cov_to_cor(mat):
    d = np.copy(np.diag(mat))
    d[d== 0] = 1
    return (1/np.sqrt(d)) *mat * (1/(np.sqrt(d)))[...,None]


N = 50
dt =.1
tstop = 2000
J =  0.1*np.random.randn(N, N)
E = 1.1
v, spktimes = sim_glm_pop(J, E, dt = dt, tstop = tstop )
print(spktimes)

raster_plot(spktimes, range(50), 0, 500)
plt.show()





start_new= time.time()
print(new_pop_correlation(spktimes,range(N), dt, tstop))
end_new = time.time()
elapsed_new = end_new - start_new


start_old= time.time()
cov_mat =  tot_cross_covariance_matrix(spktimes, range(N), dt, tstop)
cor_mat = cov_to_cor(cov_mat)
print(np.mean(cor_mat[np.triu_indices(N, 1)]))
end_old = time.time()
elapsed_old = end_old - start_old

print("new: ", elapsed_new, "old: ", elapsed_old, "ratio: ", elapsed_old/elapsed_new)