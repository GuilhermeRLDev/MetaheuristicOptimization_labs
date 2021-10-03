# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:14:33 2021

@author: Guilherme Rossetti Lima
"""
import sys
import numpy as np
import math
import random
from datetime import datetime
from os.path import join
import regex as re

#random.seed(1000)


TestDataset = "TSPdata\\TSP dataset\\test.tsp"

'''
Heuristic base class containing all the needed methods for algorithms requested
Constrctor parameters: 
    Filename: full path to the file containing the cities information
'''
class HeuristicBase():
    def __init__(self, filename):
        self.filename = filename    
        self.dictionary = self.from_file_to_array(self.filename)
        
    def from_file_to_array(self, filename):
        dictionary = dict() 
        counter = 0
        with open(filename) as file:
            counter = file.readline()
            for _ in range(int(counter)):
                row = file.readline()
                column  = row.split()
                
                dictionary[int(column[0])] = np.array([int(column[0]), float(column[1]), float(column[2])])
    
        return dictionary
    
    '''
    Calculates the euclidian distance for two points in space -- Alternatively we could have use numpy for this.        
    Parameters: 
        u: first point
        v: second point
    Returns: 
        distance: integer with lenght
    '''
    def get_weight(self, u, v):
        return int(math.sqrt(((u[1] - v[1])**2)+((u[2] - v[2])**2)))

    '''
    Returns the nearest city from in the dataset provided along with the lenght to get there
    Parameters: 
        dataset: Array init values, those values will be index from the dictionary of values
        current_city: index to city in the dictionary of cities 
    Returns: 
        chosen_city: tuple with city details
        shortest_path: integer
    '''
    def get_nearest_city(self, dataset, current_city):
        shortest_path = 0
        chosen_city = current_city
        
        '''Load location for current city from dictionary'''
        current_city_location = self.dictionary[current_city]
        
        for city in dataset:   
            city_location = self.dictionary[int(city)]
            
            distance = self.get_weight(current_city_location, city_location)
            
            if (shortest_path == 0 or distance < shortest_path):
                shortest_path = distance
                chosen_city = city 
                
        return chosen_city, shortest_path
    
'''
Implementation for Nearest neighbour heuristic
Constructor parameters: 
    filename: Array init values, those values will be index from the dictionary of values
'''
class NearestNeighbourInsertion(HeuristicBase):
    def __init__(self, filename):
        super().__init__(filename)

    '''
    Runs nearest neighbour heuristic
    Parameters: None
    Returns:
        visited: Array with indexes for cities in the dictionary, represents the final route
        lenght: Lenght for full route
    '''
    def search_solution(self):
        visited = []
        length = 0
                
        unvisited = self.dictionary.copy()
            
        ''' Picks the first city randomly '''
        random_city = random.choice(list(unvisited.keys()))
        visited.append(random_city)
        
        del unvisited[random_city]
        
        nearest = (random_city, 0)
        while (len(unvisited)>0):
            nearest = self.get_nearest_city(unvisited, nearest[0])
            visited.append(nearest[0])
            del unvisited[nearest[0]]
            
            length += nearest[1]
            
        visited.append(random_city)
        
        length += self.get_weight(self.dictionary[nearest[0]], self.dictionary[random_city])
        
        return visited, length 
'''
Implementation for alternative insertion heuristic
Constructor parameters: 
    filename: Array init values, those values will be index from the dictionary of values
'''
class  AlternativeInsertion(HeuristicBase):
    def __init__(self, filename):
        super().__init__(filename)
    
    '''
    Get nearest visited in the context of the alternative insertion, from the visited dataset (where type is list)
    Parameters: None
    Returns:
        visited: Array with indexes for cities in the dictionary, represents the final route
        lenght: Lenght for full route
    '''
    def get_nearest_visited(self, dataset, city):
        if len(dataset) == 0:
            return 0
        
        nearest_city = self.get_nearest_city(dataset, city)
        
        #Checks the index in the visited dataset, which will define where the dataset 
        return dataset.index(nearest_city[0])
    
    '''
    Runs alternative insertion heuristic 
    Parameters: None
    Returns:
        visited: Array with indexes for cities in the dictionary, represents the final route
        lenght: Lenght for full route
    '''
    def search_solution(self):
        visited = []
        length = 0
        
        ''' 
        Even though it will occupy more spare in memory for performance matters this is the 
        best way to go since a dictionary implements direct access to indexes
       '''
        unvisited = self.dictionary.copy()
        
        while(len(unvisited)>0):
            ''' Picks the a random record '''
            random_city = random.choice(list(unvisited.keys()))
            city_index = self.get_nearest_visited(visited, random_city)
            visited.insert(city_index+1, random_city)
            del unvisited[random_city]
         
        visited.append(visited[0])
        
        for i in range(len(visited)-1): 
            length += self.get_weight(self.dictionary[visited[i]], self.dictionary[visited[i+1]])  
         
        return visited, length

'''
Implementation for random heuristic
Constructor parameters: 
    filename: Array init values, those values will be index from the dictionary of values
'''
class Random(HeuristicBase):
    def __init__(self, filename):
        super().__init__(filename)
    '''
    Implements a random alg to create a route
    Parameters: None
    Returns:
        visited: Array with indexes for cities in the dictionary, represents the final route
        lenght: Lenght for full route
    '''    
    def search_solution(self):
        visited = []
        unvisited = self.dictionary.copy()
        
        random_city = random.choice(list(unvisited.keys()))
        length = 0
        visited.append(random_city)
        del unvisited[random_city]
        
        while(len(unvisited) > 0):
            random_city = random.choice(list(unvisited.keys()))
            length += self.get_weight(self.dictionary[visited[len(visited)-1]], self.dictionary[random_city])
            visited.append(random_city)
            del unvisited[random_city]
      
        length += self.get_weight(self.dictionary[visited[len(visited)-1]], self.dictionary[visited[0]])
        visited.append(visited[0])  
        
        return visited, length


'''
   Parameters: 
       heuistic_id: id for heuristic 0: Nearest, 1: Alternative, 2: Random, 3: Unit tests, 4: Analysis
       filename: path to txt file with details
   Note that for Unit tests this is not required
   Returns:
       visited: Array with indexes for cities in the dictionary, represents the final route
       lenght: Lenght for full route
''' 
def get_heuristic(heuristic_id, filename):
    
    if (heuristic_id < 0 or heuristic_id > 2 or  filename == None):
        raise Exception("Parameter provided are invalid!")
    
    
    if (heuristic_id == 0):
        return NearestNeighbourInsertion(filename)
    
    if (heuristic_id == 1):
        return AlternativeInsertion(filename)
    
    if (heuristic_id == 2):
        return Random(filename)
  
  
def run(filename, heuristic_id, outpuname: None):
    
    heuristic = get_heuristic(heuristic_id, filename)
    
    start_time = datetime.now()
    route  = heuristic.search_solution()
    end_time = datetime.now()
    
    print(route[1])
    print(route[0])
    
    print(f"Length: {route[1]}")
    print(f"Duration: {end_time  - start_time}")
    
    save_output(route,heuristic, outpuname)

def get_file_name(heuristic):
    return re.sub('[^A-Za-z0-9]+', '', str(datetime.now()))+str(type(heuristic).__name__) + ".tsp"

def save_output(route, heuristic, outputname:None):
    if (outputname == None):
        outputname = get_file_name(heuristic)
        
    with open(join("output", outputname), 'w') as file:
        file.write(f"Length: {route[1]}\r")
        print(route[0])
        for city in route[0]:
            file.write(f"{city}\r")

if __name__ == "__main__":
    
   filename, outputname, opt = TestDataset, None, 1 #Defaults to NN    
   num_params = len(sys.argv)-1
   
   if (num_params>= 1):
       filename = sys.argv[1]
       
   if (num_params >= 2):
       outputname = sys.argv[2]
       
   if (num_params >= 3):
       opt = int(sys.argv[3])
       
   '''When option is three we only run the unit tests'''
   run(filename, opt, outputname)
   
    


    
    
    
    
    