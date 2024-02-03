import ConnectAPI
import gmaps_api_opscmd

def making_weight_matrices():
    trip =
    input_string = input("Enter the places to be visited as a list of cities seperated by a space:/n")
    places = input_string.split()
    #get landmarks on paths and run weather checking on them
    #check safety information of landmarks
    graph = [[]]
    risk = [[]]
    cost = [[]]
    medium = [[]]
    gas_prices = {'AL':3.8, 'AK':3.6, 'AZ': 3.9, 'CA': 5.3, 'CO': 3.4, 'CT': 4.2, 'DC': 4.3,
                  'DE': 3.9, 'FL': 3.9, 'GA': 3.9, 'HI': 5.6, 'IA': 3.6, 'ID': 3.7,
                  'IL': 3.8, 'IN': 3.8, 'KA': 3.5, 'MA': 4.2, 'NY': 4.4, 'PA': 4.5,
                  'WI': 3.6, 'NJ': 4.0, 'RI': 4.2, 'VI': 3.9, 'WA': 4.7}
    #take into account tolls
    for i in range(len(places)):
        for j in range(len(places)):
            #find where the latitude longitudes start, then
            #(distance, time)
            if (i!=j):
                [rain, wind] = ConnectAPI.get_weather_details(LATITUDE, LONGITUDE)
                risk[i][j] = rain/2.0 + wind/30.0
                cost[i][j] = graph[i][j] * gas_prices[state] + graph[i][j]*0.15 #to account for toll and fuel money
                medium[i][j] = risk[i][j]+cost[i][j]*1.5
            else:
                risk[i][j] = 0
                cost[i][j] = 0
                medium[i][j] = 0
    return [risk, cost, medium]