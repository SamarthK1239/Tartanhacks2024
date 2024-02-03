import gmaps_api_ops as gao


def add_vertex(v):
    global graph
    global vertices_no
    global vertices
    if v in vertices:
        print("Vertex", v, "already exists")
    else:
        vertices_no = vertices_no + 1
        vertices.append(v)
        if vertices_no > 1:
            for vertex in graph:
                vertex.append(0)
        temp = [0] * vertices_no
        graph.append(temp)


# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
    global graph
    global vertices_no
    global vertices
    # Check if vertex v1 is a valid vertex
    if v1 not in vertices:
        print("Vertex", v1, "does not exist.")
    # Check if vertex v1 is a valid vertex
    elif v2 not in vertices:
        print("Vertex", v2, "does not exist.")
    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        graph[index1][index2] = e


# Print the graph
def print_graph():
    global graph
    global vertices_no
    for i in range(vertices_no):
        for j in range(vertices_no):
            if graph[i][j] != 0:
                print(vertices[i], "->", vertices[j], "edge weight:", graph[i][j])


# Driver code
# stores the vertices in the graph
vertices = []
# stores the number of vertices in the graph
vertices_no = 0
graph = []
#
# # Add vertices to the graph
# add_vertex("A")
# add_vertex("B")
# add_vertex("C")
# add_vertex("D")
#
# # Add the edges between the vertices by specifying
# # the from and to vertex along with the edge weights.
# add_edge("A", "B", 1)
# add_edge("A", "C", 1)
# add_edge("B", "C", 3)
# add_edge("C", "D", 4)
# add_edge("D", "A", 5)
#
# print_graph()
# print("Internal representation:", graph)


parsed = gao.parse_json_dictionary()
dictionary = parsed[0]
keys = list(dictionary.keys())

for origin in parsed[1][0]:
    print(origin)
    add_vertex(origin)

i = 0
for key in keys:
    # current_origin = parsed[1][0][i]
    # i += 1
    for element in [dictionary[key]]:
        # print(element["origin"])
        add_edge(element["origin"], element["destination"],
                 [element["params"]["distance"], element["params"]["duration"]])

print_graph()
print("Internal representation: ", graph)
