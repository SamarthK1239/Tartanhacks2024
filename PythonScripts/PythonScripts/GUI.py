from main_tsp_maxcut import *
import time

class Truck:
    def __init__(self, cluster, coordinates = (0, 0)):
        self.cluster = cluster
        self.position = 0
        self.location = coordinates

trucks = []
for cluster in clusters:
    trucks += [Truck(cluster)]

cities = ["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA", "New Jersey NJ", "New York NY", "Chicago IL", "Boston MA", "Allentown PA"]

def move_normal():
    global trucks
    for truck in range(len(trucks)):
        if trucks[truck].position == len(trucks[truck].cluster) - 1:
            print(f"Truck {truck} has finished journey")
        else:    
            trucks[truck].position += 1

def display_trucks():
    global trucks
    for truck in range(len(trucks)):
        print(truck, "Truck:")
        display = []
        for i in trucks[truck].cluster:
            display += [cities[i]]
        print("\t", display)
        print("\t", cities[trucks[truck].position])
    
    print("\n")


def refactor(truck_no, delay):
    truck = trucks[truck_no]
    if truck.position == len(truck.cluster) - 1:
        print("Truck has already finished journey")
        return
    else:
        edge = (truck.cluster[truck.position], truck.cluster[truck.position + 1])
    
    G[edge[0]][edge[1]]['weight'] += delay
    G[edge[1]][edge[0]]['weight'] += delay

    adjMatrix = nx.to_numpy_array(G)
    #print(adjMatrix)
    cost = 10000000
    remaining = truck.cluster[truck.position+1 :]
    for permutation in list(permutations(remaining)):
        temp_cost = adjMatrix[truck.cluster[truck.position]][permutation[0]]
        for i in range(len(permutation) - 1):
            j = i+1
            temp_cost += adjMatrix[i][j]
        if temp_cost < cost:
            cost = temp_cost
            best_order = permutation
    
    print("new order is: ", best_order, "\nand cost is ", cost)
    truck.cluster = truck.cluster[: truck.position+1] + list(best_order)
            


while True:
    display_trucks()
    move = int(input("Enter 1 for normal continuation,\nEnter 2 to disrupt,\nEnter 3 to exit:: "))
    if move == 3:
        break
    elif move == 1:
        move_normal()
    elif move == 2:
        truck_no = int(input("Which truck will have a problem? "))
        delay = int(input("What is the delay cost for this truck? "))
        refactor(truck_no, delay)
    