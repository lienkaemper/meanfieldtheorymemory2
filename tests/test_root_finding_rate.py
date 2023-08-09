import numpy as np
from src.theory import  y_pred,  C_pred_off_diag, J_eff, y_0_quad, y_corrected_quad, y_pred_from_full_connectivity,  y_corrected_quad_ein
from src.generate_connectivity import hippo_weights, gen_adjacency
import itertools 

cells_per_region = 5*np.array([1, 1, 1, 1, 1, 1])
N = np.sum(cells_per_region)
b =.5
g = 1
J = .25
h = .1

pEE = .1
pIE = .1
pII = .8
pEI = .8
import time

macro_connectivity = np.array([
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pEE, pEE, pEI, pEE, pEE, pEI],
             [pIE, pIE, pII, pIE, pIE, pII]])

A, index_dict = gen_adjacency(cells_per_region, macro_connectivity)
W =  hippo_weights(index_dict, A, h,h, g, J)

y0 = b*np.ones(N)


# y_corrected_quad(W, y0)

# y_ein = y_corrected_quad_ein(W, y0)
# y_loop = y_corrected_quad(W, y0)
# print(y_ein)
# print(y_loop)
# print(y_ein - y_loop )

def loop_sum(y_0, D, WV, Vinv, EE):
    N = WV.shape[0]
    result = np.zeros(N)
    for i in range(N):
        diff = 0
        for j, k, l, m in itertools.product(range(N), range(N), range(N), range(N)):
            diff += y_0[k] * D[i,j] * WV[j,l] * Vinv[l,k] * WV[j,m] * Vinv[m,k] * EE[l,m]
        result[i] += (1/(2*np.pi))**2 * diff 
    return result

def einsum_sum(y_0, D, WV, Vinv, EE):
    return (1/(2*np.pi))**2*np.einsum("k, ij, jl, lk, jm, mk, lm -> i", y_0,  D, WV, Vinv, WV, Vinv, EE)


import time


n = 20
y_0 = np.random.rand(n)
D = np.random.rand(n,n)
WV = np.random.rand(n,n)
Vinv = np.random.rand(n,n)
EE = np.random.rand(n,n)

start = time.time()
s1 = loop_sum(y_0, D, WV, Vinv, EE)
end = time.time()
print(end - start)

print(s1)
start = time.time()

s2 =  einsum_sum(y_0, D, WV, Vinv, EE)
end = time.time()
print(end - start)

print(s2)
print(s1 - s2)