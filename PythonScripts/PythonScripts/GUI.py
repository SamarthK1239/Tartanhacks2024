from main_tsp_maxcut import *

class Truck:
    def __init__(self, cluster, coordinates = (0, 0)):
        self.cluster = cluster
        self.position = 0
        self.location = coordinates

trucks = []
for cluster in clusters:
    trucks += [Truck(cluster)]

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
        print("\t", trucks[truck].cluster)
        print("\t", trucks[truck].position)
    
    print("\n")



while True:
    display_trucks()
    move = int(input("Enter 1 for normal continuation,\nEnter 2 to disrupt,\nEnter 9 to exit:: "))
    if move == 9:
        break
    elif move == 1:
        move_normal()
    elif move == 2:
        truck_no = int(input("Which truck will have a problem? "))
        delay = int(input("What is the delay cost for this truck? "))
        refactor()
    
