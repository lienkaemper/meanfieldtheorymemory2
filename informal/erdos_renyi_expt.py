import numpy as np 
import matplotlib.pyplot as plt

N = 20
M = 20
T = 5 
p = 0.4

fig, ax = plt.subplots()
for i in range(T):
    A = np.random.rand(N, N) < p
    A = A.astype(int)
    print(A)
    B = np.eye(N)
    Bs = [B]
    for m in range(M):
        B = B @ A
        Bs.append(B)

    print(B)

    for m, B in enumerate(Bs):
        pred_val = N**(m-1) * p**(m) 
        print(pred_val)
        ax.scatter(x = pred_val*np.ones(N**2), y = Bs[m].flatten())
        ax.set_xlabel("pred")
        ax.set_ylabel("sim")
        ax.set_xscale("log")
        ax.set_yscale("log")




x = np.linspace(0, N**M * p**(m-1))
ax.plot(x,x)
plt.show()