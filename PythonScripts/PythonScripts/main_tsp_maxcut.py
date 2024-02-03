from TSP import *
from max_cut import *

n = int(input("Number of trucks in fleet:: "))
numberoftrucks = n
clusters = []

def main(graph):    
    global numberoftrucks, clusters
    clusters += [list(graph.nodes())]

    while len(clusters) < numberoftrucks:
        biggest = biggest_cluster(clusters)
        biggest = clusters.pop(biggest)

        split, nodes = main_max_cut(graph.subgraph(biggest))
        left, right = [], []
        for i in range(len(split)):
            if split[i] == 0:
                left += [nodes[i]]
            else:
                right += [nodes[i]]
        clusters += [right]
        clusters += [left]
    
    for i in clusters:
        yield main_tsp(graph.subgraph(i))

def biggest_cluster(clusters):
    biggest = 0
    for i in range(len(clusters)):
        if len(clusters[i]) > len(clusters[biggest]):
            biggest = i
    
    return biggest

G = nx.complete_graph(6)
for i in range(5):
    for j in range(i+5, 5):
        G[i, j]["weight"] = 5
        
vertices = [0, 1, 2, 3, 4]
edges = [(0, 1), (0, 3), (0, 4), (1, 2), (2, 3), (4, 1)]
G = nx.Graph()
G.add_nodes_from(vertices)
for edge in edges:
    G.add_edge(edge[0], edge[1], weight = 3)
print(nx.to_numpy_array(G))

print(list(main(G)))