import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.generate_connectivity import excitatory_only


p = 0.1
N_engram = 25
N = 4*N_engram 
h = 3
J0 = 0.25

A = np.random.rand(4*N_engram, 4*N_engram) < p 
index_dict = {"CA3E": range(N_engram), "CA3P": range(N_engram, 2*N_engram), "CA1E": range(2*N_engram, 3*N_engram), "CA1P": range(3*N_engram, 4*N_engram)}
J = excitatory_only(index_dict, A, h, J0)

pos = np.zeros((N, 2))
pos[index_dict["CA3E"], :] += [0,1]
pos[index_dict["CA3P"], :] += [1,1]
pos[index_dict["CA1P"], :] += [1,0]
pos += 0.6*np.random.rand(N,2)

G = nx.from_numpy_array(J.T, create_using = nx.DiGraph)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
print(weights)
pos = dict(zip(nx.nodes(G), pos))
node_ids = np.zeros(N, np.int8)
node_ids[index_dict["CA3E"]] += 1
node_ids[index_dict["CA1E"]] += 1
nx.draw_networkx_nodes(G, pos, node_size = 50, node_color=node_ids)
nx.draw_networkx_edges(G, pos, edgelist = edges, width = weights)
plt.gca().set_aspect('equal')
plt.savefig("../results/graph_cartoon_no_inhib.pdf")
plt.show()
