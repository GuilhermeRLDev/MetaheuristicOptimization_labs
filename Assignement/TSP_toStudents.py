

"""
Author: Guilherme Rossetti Lima
file:
Rename this file to Lastname_Firstname_StudentNumber.py
"""

import random
from Individual import *
import sys
import math
import queue
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from os.path import join
from os.path import exists
from os import mkdir

myStudentNum = 209056 
random.seed(myStudentNum)

DEFAULT_PATH_4  = "D:\\Master\\MetaheuristicOptimization\\Lab1\\TSPdata\\TSP dataset\\inst-4.tsp"
DEFAULT_PATH_16  = "D:\\Master\\MetaheuristicOptimization\\Lab1\\TSPdata\\TSP dataset\\inst-16.tsp"
DEFAULT_PATH_6  = "D:\\Master\\MetaheuristicOptimization\\Lab1\\TSPdata\\TSP dataset\\inst-6.tsp"
OUTPUTS_PATH_DIR = "outputs"
EXECUTION_PLOTS_DIR = "execution_plots"
GENERATION_PLOTS_DIR = "generation_plots"

class HeuristicBase():
    def __init__(self, data):
        self.data = data
    
    '''
    Calculates the euclidian distance for two points in space
    Parameters: 
        u: first point
        v: second point
    Returns: 
        distance: integer with lenght
    '''
    def get_distance(self, u, v):
        return int(math.sqrt(((u[0] - v[0])**2)+((u[1] - v[1])**2)))
    
    '''
    Returns the nearest city in the dataset provided along with its distance from the current node
    Parameters: 
        unvisited: Array  - contains the init values with the index for the unvisited cities
        current_city: index to city in the dictionary of cities 
    Returns: 
        chosen_city: tuple with city details
        shortest_path: integer
    '''
    def get_nearest_city(self, unvisited, current_city):
        shortest_path = 0
        chosen_city = current_city
        
        '''Load location for current city from dictionary'''
        current_city_location = self.data[current_city]
        
        for city in unvisited:   
            city_location = self.data[int(city)]
            
            distance = self.get_distance(current_city_location, city_location)
            
            if (shortest_path == 0 or distance < shortest_path):
                shortest_path = distance
                chosen_city = city 
                
        return chosen_city, shortest_path

class NearestNeighbourInsertion(HeuristicBase):
    def __init__(self, data):
        super().__init__(data)
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
                
        unvisited = self.data.copy()
            
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
        
        length += self.get_distance(self.data[nearest[0]], self.data[random_city])
        
        return visited 

class BasicTSP:
    def __init__(self, _fName, _maxIterations, _popSize, _initPop, _xoverProb, _mutationRate, _trunk, _elite):
        """
        Parameters and general variables
        Note not all parameters are currently used, it is up to you to implement how you wish to use them and where
        """
        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = int(_popSize)
        self.genSize        = None
        self.crossoverProb   = float(_xoverProb)
        self.mutationRate   = float(_mutationRate)
        self.maxIterations  = int(_maxIterations)
        self.trunkSize      = float(_trunk) 
        self.eliteSize      = float(_elite) 
        self.fName          = _fName
        self.initHeuristic  = int(_initPop)
        self.iteration      = 0
        self.data           = {}
        self.sortedPool = None
        self.best_per_generation = None
        #stats for telemetrics
        self.best_initial = None
        self.best_solution = None
        self.generation_number = 0
        self.average_fitness = 0
        self.average_fitness_generation = 0
        self.initialization_runtime_cost = 0
        
        
        self.readInstance()
        self.initPopulation()
        print(self.popSize,self.crossoverProb,self.mutationRate,self.maxIterations,self.trunkSize,self.eliteSize)


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        start = datetime.now()
        
        # Create NN insertion heuristic in case it is initiliaxation mode
        nnHeuristic  = NearestNeighbourInsertion(self.data)
        
        for i in range(0, self.popSize):
            indGenes =[]
            if (self.initHeuristic == 1):
                indGenes = nnHeuristic.search_solution()
                
            individual = Individual(self.genSize, self.data, indGenes)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
                self.best_per_generation = ind_i.copy()
                # keep it to log telemetry
                self.best_initial = ind_i.getFitness() 
        end = datetime.now()       
        
        #Save initialization cost for telemetry
        self.initialization_runtime_cost = (end - start).total_seconds() 
        
        print ("Best initial sol: ",self.best.getFitness())

    def updateBest(self, candidate, generation):
        
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            # Save the generation where the best was found for telemetry
            self.generation_number = generation 
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())
            
        if self.best_per_generation == None or candidate.getFitness() < self.best_per_generation.getFitness():
            self.best_per_generation = candidate.copy()
    
    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def truncationTournamentSelection(self):
        """
        Your Truncation Tournament Selection Implementation to fill the mating pool
        """
        
        if (len(self.matingPool) == 0):
            self.matingPool = []
            sortedPool = queue.PriorityQueue()
            # Sorts the pool using a priority queue
            i= 1
            for individual in self.population:
                sortedPool.put(individual)
                
            i= 1
            while ((i/len(self.population)) < self.trunkSize):
                ind = sortedPool.get()
                self.matingPool.append(ind.genes)
                i += 1
                
            #random.shuffle(self.matingPool) #To keep fairness netween the fittest we shuffle the array
        
        indA = self.matingPool[random.randint(0, len(self.matingPool)-1)]
        indB = self.matingPool[random.randint(0, len(self.matingPool)-1)]
        
        return indA, indB
    
    def order1Crossover(self, indA, indB):
        '''Pick a random start and end point'''
        point1 = random.randint(0, len(indA))
        point2 = random.randint(point1, len(indA))

        '''Selects genes from individual A based on random selection'''
        randomGenesFromA = indA[point1:point2]
        
        dictA = dict.fromkeys(indA, 0)
        dictB = dict.fromkeys(indB, 0)
        
        for a in randomGenesFromA:
            del dictB[a]
            
        cgenes_a = list(dictB.keys()) + randomGenesFromA
        
        '''Selects genes from individual B based on random selection'''
        randomGenesFromB = indB[point1:point2]
        
        for b in randomGenesFromB:
            del dictA[b]
       
        '''Generates final child genes by concatenating cgenes and random selected genes from A'''
        cgenes_b = list(dictA.keys()) + randomGenesFromB
        
        ''' Returns new individual'''
        return Individual(len(cgenes_a), self.data, cgenes_a), Individual(len(cgenes_b), self.data, cgenes_b)

    def inversionMutation(self, ind):
        
        if (random.random() > self.mutationRate):
            return
        
        '''Randomly selects two points'''
        point1 = random.randint(0, len(ind.genes))
        point2 = random.randint(point1, len(ind.genes))
        
        '''slices region and update it  in the individual genes by the inverted version'''
        slicedRegion = ind.genes[point1:point2]
        ind.genes[point1:point2] = slicedRegion[::-1]
        
    
    def getChildren(self): 
        parent1, parent2 = self.truncationTournamentSelection()
        
        if (random.random() < self.crossoverProb):
            child1, child2 = self.order1Crossover(parent1,parent2) 
        else:
            child1 = Individual(len(parent1), self.data, parent1)
            child2 = Individual(len(parent2), self.data, parent2)
        
        self.inversionMutation(child1)
        self.inversionMutation(child2)
        
        child1.computeFitness()
        child2.computeFitness()
        
        return child1, child2
 
    def getElite(self, eliteSize):
        elite = []
        for i in range(eliteSize):
            if (not self.elite.empty()):
                ind = self.elite.get()
                elite.append(ind)
        
        self.elite = queue.PriorityQueue()
        
        return elite
 
    def newGeneration(self, iteration):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        new_population = []
        for i in range(int(self.popSize/2)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            parent1, parent2 = self.truncationTournamentSelection()
            
            
            
            self.updateBest(child1, iteration)
            self.updateBest(child2, iteration)
            
            new_population.append(child1)
            new_population.append(child2)
            
            self.average_fitness_generation += child1.getFitness()
            self.average_fitness_generation += child2.getFitness()
            
        self.average_fitness_generation = self.average_fitness_generation / self.popSize
        
        self.population = new_population
    
    def newGenerationWithElitismPX(self, iteration):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        elitismSize = int(self.eliteSize * self.popSize)

        elite = self.getElite(elitismSize) 
        new_population = []
        
        for i in range(int((self.popSize-elitismSize)/2)):
            child1, child2 = self.getChildren()
            
            self.updateBest(child1, iteration)
            self.updateBest(child2, iteration)
            
            new_population.append(child1)
            new_population.append(child2)
            
        self.population =  new_population + elite
    
        for ind in self.population:
            self.average_fitness_generation += ind.getFitness()
            self.elite.put(ind)
        
        self.average_fitness_generation = self.average_fitness_generation / self.popSize
        
    '''
    This method contains the implementation for elitims, when elitSize equals to zero n elitsm is applied
    '''
    def newGeneration(self, iteration):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
        elitismSize = int(self.eliteSize * self.popSize)

        new_population = []
        top = queue.PriorityQueue()
        bottom = queue.PriorityQueue()
        elite = queue.PriorityQueue()
        
        #Generates full population
        for i in range(int(self.popSize/2)):
            child1, child2 = self.getChildren()
            
            self.updateBest(child1, iteration)
            self.updateBest(child2, iteration)
            
            bottom.put(child1)
            bottom.put(child2)
        
        #Order populatioin by fitness
        for ind in self.population:
            top.put(ind)
        
        #Populate the new population with the top items generate in the new population
        for i in range(self.popSize - elitismSize):
            if not bottom.empty():
                ind = bottom.get()
                new_population.append(ind) 
                #Populate the average
                self.average_fitness = ind.getFitness()

        #Populate the elite queue from bottom and top by fitness
        for i in range(elitismSize):
            if not top.empty():
                ind = top.get()
                elite.put(ind) 
                
            if not bottom.empty():
                ind = bottom.get()
                elite.put(ind) 
        
        #Feed top x fittest into the new population
        for i in range(elitismSize):
            if not elite.empty():
                ind = elite.get()
                new_population.append(ind) 
                #Populate the average
                self.average_fitness_generation += ind.getFitness()
        
        self.average_fitness_generation = self.average_fitness_generation / self.popSize
        
        # Replacement
        self.population = new_population
    
        
    def GAStep(self, iteration):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """
        self.matingPool = []
        self.newGeneration(iteration)

    def get_mean(self, x, y):
        total = 0
        averages = []
        i = 0
    
        for value in y:
            total += value
            averages.append(total/(x[i]+1))
            i += 1
        
        return averages
    
    def plot(self, best_per_generation, runs, title, file_name):
        fig, ax = plt.subplots()
           
        x = np.arange(0, runs)
           
        '''Plot nearest stats'''
        nearest = ax.plot(x, best_per_generation, label="Fitness")
        nearest_mean = ax.plot(x, self.get_mean(x, best_per_generation), label="Average", linestyle='--')
           
        legend = ax.legend(loc='upper right')
        try:
            plt.title(title, loc='center')
            plt.savefig(join(GENERATION_PLOTS_DIR, file_name))
        except:
            print("Error plotting generation graph")
        
    def search(self, file_name=None):
        """
        General search template
        Iterates for a given number of steps
        """
        self.iteration = 0
        best_per_iteration = []
        while self.iteration < self.maxIterations:
            self.average_fitness_generation = 0
            self.GAStep(self.iteration+1)
            best_per_iteration.append(self.best_per_generation.getFitness())
            self.best_per_generation = None
            self.iteration += 1
            
            self.average_fitness += self.average_fitness_generation
            
        
        # Save average fitness for telemetry
        self.average_fitness = self.average_fitness / self.maxIterations
        
        if file_name:
            self.plot(best_per_iteration, self.iteration, "Best fitness per generation", file_name)    
        
        # Commented out as printing the value in the runs may affect the overall runtime cost
        #for ind in self.population:
        #    print(ind.getFitness())
        
        print ("Total iterations: ", self.iteration)
        print ("Best Solution: ", self.best.getFitness())


def get_mean(x, y):
    total = 0
    averages = []
    i = 0
    
    for value in y:
        total += value
        averages.append(total/(x[i]+1))
        i += 1
        
    return averages

def plot(generation, runs, title, leged, file_name):
        fig,ax = plt.subplots()
           
        x = np.arange(0, runs)
           
        '''Plot nearest stats'''
        nearest = ax.plot(x, generation, label=leged[0])
        nearest_mean = ax.plot(x, get_mean(x, generation), label=leged[1], linestyle='--')
           
        legend = ax.legend(loc='upper right')
        try:
            plt.title(title, loc='center')
            plt.savefig(join(EXECUTION_PLOTS_DIR, file_name))
            plt.show()
        except:
            print("Error ploting metrics!")

def save_stats_per_run(ga, experiment_name):
    with open(join(OUTPUTS_PATH_DIR, f"{experiment_name}.csv"), "a", encoding="UTF8") as file:
        file.write(f"{ga.best_initial},{ga.best.getFitness()},{ga.generation_number},{ga.average_fitness},{ga.initialization_runtime_cost}\n")

#if len(sys.argv) < 9:
#    print ("Error - Incorrect input")
#    print ("Expecting python TSP.py [instance] [number of runs] [max iterations] [population size]", 
#           "[initialisation method] [xover prob] [mutate prob] [truncation] [elitism] ")
#    sys.exit(0)

if (not exists(OUTPUTS_PATH_DIR)):
    mkdir(OUTPUTS_PATH_DIR)
    
if (not exists(EXECUTION_PLOTS_DIR)):
    mkdir(EXECUTION_PLOTS_DIR)    

if (not exists(GENERATION_PLOTS_DIR)):
    mkdir(GENERATION_PLOTS_DIR)

def run(f, inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP, file_name="experiment"):    
    bestPerRun = []
    executionTime = []
    
    for run in range(nRuns):
        ga = BasicTSP(inst, nIters, pop, initH, pC, pM, trunkP, eliteP)
        
        #Execute the search and record best result and execution time
        start = datetime.now()
        ga.search(f"{file_name}-run-{run}")
        bestPerRun.append(ga.best.getFitness())
        end = datetime.now()
        timElapsed  = end - start
        executionTime.append(timElapsed.total_seconds())
        
        #Save stats in file
        save_stats_per_run(ga, f"{file_name}")
    
    plot(bestPerRun, nRuns,"Best fitness per run", ["Finess", "Average"], f"{file_name}-runtime" )
    plot(executionTime, nRuns, "Runtime cost per run ", ["Exe. time", "Average"], f"{file_name}-executioncost")

if (len(sys.argv) < 9):
    f, inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP = None, DEFAULT_PATH_6, 20, 500, 100, 0, 0.0, 0.05, 0.25, 0.1
else:
    f, inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP = sys.argv

run(None, DEFAULT_PATH_4, 10, 500, 100, 1, 0.8, 0.005, 0.25, 0.1,"mutation-05-NN-inst-4")
run(None, DEFAULT_PATH_4, 10, 500, 100, 1, 0.8, 0.05, 0.25, 0.1,"mutation-5-NN-inst-4")
run(None, DEFAULT_PATH_4, 10, 500, 100, 1, 0.8, 0.5, 0.25, 0.1,"mutation-50-NN-inst-4")




'''Impact of population size probability inst 16
run(None, DEFAULT_PATH_4, 10, 500, 10, 1, 0.8, 0.05, 0.25, 0.0,"population-impact-NN-inst-16")
run(None, DEFAULT_PATH_4, 10, 500, 100, 1, 0.8, 0.05, 0.25, 0.1,"population-impact-NN-inst-16")
run(None, DEFAULT_PATH_4, 10, 500, 1000, 1, 0.8, 0.05, 0.25, 0.1,"population-impact-NN-inst-16")


run(None, DEFAULT_PATH_4, 10, 500, 10, 1, 0.8, 0.05, 0.25, 0.0,"population-impact-NN-inst-6")
run(None, DEFAULT_PATH_4, 10, 500, 100, 1, 0.8, 0.05, 0.25, 0.1,"population-impact-NN-inst-6")
run(None, DEFAULT_PATH_4, 10, 500, 1000, 1, 0.8, 0.05, 0.25, 0.1,"population-impact-NN-inst-6")
'''

'''
Reading in parameters, but it is up to you to implement what needs implementing
e.g. truncation selection, elitism, initialisation with heuristic vs random, etc'''

'''
bestPerRun = []
executionTime = []
for run in range(nRuns):
    ga = BasicTSP(inst, nIters, pop, initH, pC, pM, trunkP, eliteP)
    #Execute the search and record best result and execution time
    start = datetime.now()
    ga.search()Best initial sol:  21373660.442692492
    bestPerRun.append(ga.best.getFitness())
    end = datetime.now()
    timElapsed  = end - start
    executionTime.append(timElapsed.total_seconds())
    
plot(bestPerRun, nRuns,"Best fitness per run", ["Finess", "Average"])
plot(executionTime, nRuns, "Runtime cost per run ", ["Exe. time", "Average"])
'''  
    
