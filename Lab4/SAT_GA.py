# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 21:44:00 2021

@author: Guilherme Rossetti Lima
"""
import random  

class SATVerifier:
    def __init__(self, variables, clauses):
        self.variables = variables
        self.clauses = clauses
    
    def get_index(self, variable):
        return int(str.replace(variable, '-', ''))
    
    def is_negation(self, value):
        return str.find(value, "-") > -1 
    
    def verify_clause(self, clause): 
        satistied = False
        for variable in clause:
            if (self.is_negation(variable)):
                satistied = satistied or not self.variables[self.get_index(variable)]
            else:
                satistied = satistied or self.variables[self.get_index(variable)]
        
        return satistied
    
    def verify_solution(self):
        satisfied = True
        
        for clause in self.clauses:
            satisfied = satisfied and self.verify_clause(clause)
            
            if (not satisfied):
                return satisfied
        
        return satisfied
   
    
class CNFReader: 
    def __init__(self):
        self.variables = dict()
        self.clauses = []
        
        
    def interpret_problem(self, line):
        items = str.split(line, " ")
    
        self.num_variables = int(items[2])        
        self.num_clauses = items[4]    

    def process_line(self, line):
        line = str.strip(line)
        
        if(line[0] == "c"):
            return
    
        if(line[0] == 'p'):
            self.interpret_problem(line)
            return
        
        variables = str.split(line, " ")
        self.clauses.append(variables[:len(variables)-1])
       
        for variable in variables:
            if self.get_index(variable) not in self.variables:
                self.variables[self.get_index(variable)] = False
    
    def get_index(self, variable):
        return int(str.replace(variable, '-', ''))
        
    def read_input_file(self, file_name):
        with open(file_name, "r") as file:
            line = file.readline()
            while line != "%\n":
                self.process_line(line)
                
                line = file.readline()
                
        return self.variables, self.clauses
    
   
class SATGA:
    '''
    Creates the cromossomes based in the variables and clauses
    Parameters: 
        variables: all variables
        clauses: clauses
    '''
    def get_random_individual(self, size):
        return [random.choice([True,False]) for _ in range(size)]
    
    def __init__(self, mutation_rate, max_population, variables, clauses):
        self.mutation_rate = mutation_rate
        self.max_population = max_population
        self.variables = variables
        self.clauses = clauses
        self.population = [self.get_random_individual(len(variables)) for _ in range(max_population)]
        self.matingPool = []
        self.best_fitness = 0
        self.best = None
    
    def step(self):
        mating_pool = [] 
        
        self.best = 0
        
        for individual in self.population:
            numberOfCopies = int(self.getFitness(individual))
            
            for _ in range(0, numberOfCopies):
                mating_pool.append(individual)
        
        for i in range(0, len(self.population)):
            
            parent1 = mating_pool[random.randint(0, len(mating_pool)-1)] 
            parent2 = mating_pool[random.randint(0, len(mating_pool)-1)]
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            fitness = self.getFitness(child)
            
            self.population[i] = child
            
            if (self.best_fitness < fitness):
                self.best = child
                self.best_fitness = fitness
                
                print(f"Best fit so far: {fitness}")
                print(self.best)
                print(self.get_variables(self.best))
       
    def get_variables(self, values):
        keys = list(self.variables.keys())
        variables = dict.fromkeys(self.variables.keys(), False)
        
        for index in range(len(values)):
            variables[keys[index]] = values[index]
            
        return variables
            
    def getFitness(self, individual):
        total_satisfied = 0
        
        verifier = SATVerifier(self.get_variables(individual), self.clauses)
        for clause in self.clauses:
            if (verifier.verify_clause(clause)):
                total_satisfied += 1
        
        return (total_satisfied / len(self.clauses)) * 100
   
    def crossover(self, ind1, ind2):
        child = []
        
        split_point = random.randint(0, len(self.variables))
        child[0: split_point] = ind1[0: split_point]
        child[split_point: len(self.variables)] = ind2[split_point: len(self.variables)]
        
        return child
    
    def mutate(self, individual):
        for i in range(len(individual)):
            if (random.random() < self.mutation_rate):
                individual[i] = not individual[i]
        
        return individual
                
    def search(self, number_interations):
        
        i = 0
        while(i < number_interations and self.best_fitness < 100): 
            i+= 1
            self.step()
                
if __name__  == "__main__": 
    variables, clauses = CNFReader().read_input_file("D:\\Master\\MetaheuristicOptimization\\MetaheuristicOptimization_labs\\Lab2\\Lab-data\\Inst\\uf20-02.cnf")
    ga = SATGA(0.1, 10000, variables, clauses)
    
    ga.search(5000)
    
    print(ga.best_fitness)
    print(ga.best)


