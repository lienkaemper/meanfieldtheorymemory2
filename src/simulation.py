import numpy as np 
from tqdm import tqdm 

def intensity(v, B=1, v_th=1, p=1):
    x = v - v_th 

    if len(np.shape(x)) > 0:
        x[x < 0] = 0
    elif x < 0:
        x = 0
    else: pass
    
    return B * x**p

def sim_glm_pop(J, E, tstop=100, dt=.01, B=1, v_th=1, p=1, v_r=0, tstim=None, Estim=0, v0=0, maxspikes = 2*10**6):
    if tstim is None:
        tstim = tstop
    Nt = int(tstop / dt)
    Ntstim = int(tstim / dt)

    if len(np.shape(J)) > 1:
        N = np.shape(J)[0]
    else:
        N = 1
    
    if len(np.shape(E)) == 0:
        E = E * np.ones(N,)
    elif len(E) < N:
        raise Exception('Need either a scalar or length N input E')

    v = np.zeros((Nt,N), np.float32)
    v[0] = v0
    
    n = np.zeros(N,np.float16)
    spktimes = np.empty((maxspikes,2), np.float64)
    nspikes = 0
    for t in tqdm(range(1, Nt)):

        if t > Ntstim:
            Et = Estim
        else:
            Et = E

        v[t] = v[t-1] + dt*(-v[t-1] + Et) + np.dot(J, n)

       # lam = intensity(v[t], B=B, v_th=v_th, p=p)
        lam = intensity(v[t], B=B, v_th=v_th, p=p)
        lam[lam > 1/dt] = 1/dt
            
        n = np.random.binomial(n=1, p=dt*lam) #note: this is /not/ poisson. does it matter? (note that it being poisson is weird -- since we only allow at most one spike per time bin, and the poisson distribution would allow multiple spikes)

        spkind = np.where(n > 0)[0]
        for i in spkind:
            if nspikes <maxspikes:
                spktimes[nspikes, :] = [t*dt, i]
                nspikes += 1

            
    spktimes = spktimes[0:nspikes, :]
    return v, spktimes

    