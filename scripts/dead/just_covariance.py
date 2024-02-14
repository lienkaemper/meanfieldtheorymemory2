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