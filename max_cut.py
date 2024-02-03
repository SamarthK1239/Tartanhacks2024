import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit_optimization.applications import Maxcut
from qiskit_algorithms import NumPyMinimumEigensolver

from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Generating a graph of 4 nodes

n = 5  # Number of nodes in graph
G = nx.MultiDiGraph()
G.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0), (4, 2, 5.0), (4, 1, 1.0), (4, 0, 1.0)]
# tuple is (i,j,weight) where (i,j) is the edge
G.add_weighted_edges_from(elist)

colors = ["r" for node in G.nodes()]
pos = nx.spring_layout(G)


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


#draw_graph(G, colors, pos)

# Computing the weight matrix from the random graph
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        print(type(temp))
        if temp != 0:
            w[i, j] = temp["weight"]

max_cut = Maxcut(w)
qp = max_cut.to_quadratic_program()

qubitOp, offset = qp.to_ising()

exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(qp)
print(result.fval)
for i in result:
    print(i)

