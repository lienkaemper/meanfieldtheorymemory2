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

def create_spike_train_matrix(spktimes, neurons, dt=.01, tstop=100):
    
    '''
    create a spike train from a list of spike times and neuron indices
    spktimes: Nspikes x 2, first column is times and second is neurons
    dt and tstop should match the simulation that created spktimes
    '''
    Nt = int(tstop/dt)+1
    n = len(neurons)
    spktrain=np.zeros((n, Nt))

    for i, n_i in enumerate(neurons):
        spktimes_tmp = spktimes[spktimes[:, 1] == n_i][:, 0]
        spk_indices = spktimes_tmp / dt
        spk_indices = spk_indices.astype('int')
        spktrain[i, spk_indices] = 1/dt

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
            if i <= j:
                C[c_i,c_j] = tot_cross_covariance(spktimes,i,j,dt,tstop)
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

def rates(spktimes, neurons, dt=.01, tstop=100):
    return np.array([len(spktimes[spktimes[:, 1] == neuron][:, 0])/tstop for neuron in neurons])

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
    
# if C is a vector, computes mean over each region as specified in index dict, returns vector
# if C is matrix, computes mean of off-diagonal elements over each region as specified in index dict, returns matrix 
def sum_by_region(J, index_dict):
    N_regions = len(index_dict)
    Ns = [len(index_dict[region]) for region in index_dict]
    if J.ndim == 2:
        J_sum = np.zeros((N_regions, N_regions))
        for i, region_i in enumerate(index_dict):
            for j, region_j in enumerate(index_dict):
                J_loc =J[np.ix_(index_dict[region_i], index_dict[region_j])]
                J_sum[i,j] = np.sum(J_loc)/Ns[i]
        return J_sum


def mean_pop_correlation(spktimes, neurons, dt, tstop):
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
