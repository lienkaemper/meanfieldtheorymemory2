import numpy as np
from scipy import signal

def create_spike_train(spktimes, neuron=0, dt=.01, tstop=100):
    
    '''
    create a spike train from a list of spike times and neuron indices
    spktimes: Nspikes x 2, first column is times and second is neurons
    dt and tstop should match the simulation that created spktimes
    '''
    
    spktimes_tmp = spktimes[spktimes[:, 1] == neuron][:, 0]
    
    Nt = int(tstop/dt)+1
    spktrain = np.zeros((Nt,))
    
    spk_indices = spktimes_tmp / dt
    spk_indices = spk_indices.astype('int')

    spktrain[spk_indices] = 1/dt
    
    return spktrain

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
            C[c_i,c_j] = tot_cross_covariance(spktimes,i,j,dt,tstop)
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


def tot_pop_autocovariance(spktimes, neurons, dt = .01, tstop = 100):
    spk = create_pop_spike_train(spktimes, neurons, dt, tstop)
    spk -= np.mean(spk)
    _, Ctmp = signal.csd(spk, spk, fs = 1/dt, scaling = 'density', window = 'bartlett', nperseg = 2048, return_onesided = False, detrend = False)
    return Ctmp[0]


def two_pop_covariance(spktimes, pop1, pop2, dt = .01, tstop = 100):
    spk1 = create_pop_spike_train(spktimes, pop1, dt, tstop)
    spk1 -= np.mean(spk1)
    spk2 = create_pop_spike_train(spktimes, pop2, dt, tstop)
    spk2 -= np.mean(spk2)
    _, Ctmp = signal.csd(spk1, spk2, fs = 1/dt, scaling = 'density', window = 'bartlett', nperseg = 2048, return_onesided = False, detrend = False)
    return Ctmp[0]

def rate(spktimes, neuron=0, dt=.01, tstop=100):
    spktimes_tmp = spktimes[spktimes[:, 1] == neuron][:, 0]
    return len(spktimes_tmp)/tstop

def pop_rate(spktimes, neurons, dt=.01, tstop=100):
    spktimes_tmp = create_pop_spike_train(spktimes, neurons, dt, tstop)
    return sum(spktimes_tmp)*dt/tstop


# if C is a vector, computes mean over each region as specified in index dict, returns vector
# if C is matrix, computes mean of off-diagonal elements over each region as specified in index dict, returns matrix 
def mean_by_region(C, index_dict):
    N_regions = len(index_dict)
    Ns = [len(index_dict[region]) for region in index_dict]
    if C.ndim == 1:
        C_mean = np.zeros(N_regions)
        for i, region_i in enumerate(index_dict):
            C_local =C[index_dict[region_i]]
            C_mean[i] = np.mean(C_local)
        return C_mean
    if C.ndim == 2:
        C_mean = np.zeros((N_regions, N_regions))
        for i, region_i in enumerate(index_dict):
            for j, region_j in enumerate(index_dict):
                C_local =C[np.ix_(index_dict[region_i], index_dict[region_j])]
                C_mean[i,j] = np.sum(C_local)
                if i == j:
                    C_mean[i,j] -= np.sum(np.diag(C_local))
                    C_mean[i, j] /= Ns[i]*(Ns[i]-1)
                else:
                    C_mean[i, j]/= Ns[i]*Ns[j]
        return C_mean
    