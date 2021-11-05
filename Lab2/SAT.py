# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:02:41 2021

@author: Guilherme Rossetti Lima
"""
import sys
import io
from os import listdir
from os.path import join, isfile
from collections import deque
import random

INPUT = "D:\\Master\\MetaheuristicOptimization\\MetaheuristicOptimization_labs\\Lab2\\Lab-data\\Inst\\uf20-04.cnf"
OUTPUT = "D:\\Master\\MetaheuristicOptimization\\MetaheuristicOptimization_labs\\Lab2\\Lab-data\\sols\\"

class Problem:
    def __init__(self, actions, variables):
        self.variables = list(variables.keys())
        
        self.num_expantions = 0
    
    def actions(self, state):
        if (len(self.variables) == state):
            return []
        
        return [True, False]
    
    def result(self, state, action):
        return self.variables[state+1]
        
    def goal_test(self, goal_test, node):
        test = goal_test(node, self)
        self.num_expantions += 1
        return test
    
class Node:
     def __init__(self, state, action, parent):
         self.state = state
         self.action = action 
         self.parent = parent 
         
     def expand(self, problem):
         nodes =  [self.child_node(problem, action) for action in problem.actions(self.state)]
         
         return nodes
     
     def child_node(self, problem, action):
         state = problem.result(self.state, action)
         
         return Node(state, action, self)
     
     def path(self, problem):   
         variables = dict([])
         
         for i in range(1, len(problem.variables)+1):
             variables[i] = False
         
         c_node = self
         while(c_node.parent != None):
             variables[c_node.state] = c_node.action
             c_node = c_node.parent
             
         return variables
            
     
class SATVerifier():
    def __init__(self, input_file, output_file=None): 
        self.input_file = input_file
        self.output_file = output_file
        self.num_variables = 0
        self.num_clauses = 0
        
        self.clauses = []
        
        self.variables = dict()
        
        self.read_input_file()
        
        if (output_file):
            self.read_output_file()
        
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
        
    def is_negation(self, value):
        return str.find(value, "-") > -1 
    
    def get_index(self, variable):
        return int(str.replace(variable, '-', ''))

    def process_variable_values(self, line):
        line = str.strip(line)
        variables = str.split(line, " ")
        
        for variable in variables:
            self.variables[self.get_index(variable)] = not self.is_negation(variable)
           
    def process_output_line(self, line):
        
        if(line[0] == "c"):
            return
    
        if(line[0] == 'v'):
            self.process_variable_values(line[3:-1])
        
    def read_input_file(self):
        with open(self.input_file, "r") as file:
            line = file.readline()
            while line != "%\n":
                self.process_line(line)
                
                line = file.readline()
                
    def read_output_file(self):
        with open(self.output_file, "r") as file:
            line = file.readline()
            while line != "v  0" and line != "":
                self.process_output_line(line)
                
                line = file.readline()  
        
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
    
    def seed_variables(self, value):
        for i in range(1, self.num_variables + 1):
            self.variables[i] = value
    
    
    def is_fully_satisfied_by_variable(self, literal, value, clause):
        for item in clause:
            if (self.get_index(item) == self.get_index(literal) and 
                ((self.is_negation(item) and value) or 
                 (not self.is_negation(item) and not value))):
                print(f"Item{item}")
                return False
            
        return True

    def goal_function(self, node, problem):
        self.variables = node.path(problem)
        return self.verify_solution() 

    def binary_tree_search(self):
        self.seed_variables(False)
        problem = Problem([True, False], self.variables)
        initial = Node(0, False, None)
        
        frontier = deque([initial])
        
        while len(frontier) > 0:
            node = frontier.pop()
            
            if(problem.goal_test(self.goal_function, node)):
                print("Found goal")
                return node
            
            frontier.extend(node.expand(problem))
            
            if (problem.num_expantions <51):
                print(self.variables)
        
        print(problem.num_expantions)
    
    '''
    Create a queue of unsatistied clauses (FIFO).
    Clone variables list.
    Create a list of variables to watch.
    Pick a in the first clause.
    Check if that is the list of variables to watch, if so check if clause is satisfied. 
    If not safisfied move to next literal, repeat the check.
    If variable not in the watch list set it in a way to satisfy the clause(keep an eye out for negations).
    Set the literal in such a way you satisfy the clause.
    Remove from list of unsatisfied clauses clauses.
    Remove clause from the queue.
    '''
    def watch_list_heuristic(self):
        
        unsatisfied_clauses = deque(self.clauses)
        unsatisfiable_clauses = deque([])
        
        print(self.clauses)
        
        unset_variables = self.variables.copy()
        watch_variables = dict()
        
        while ((len(unset_variables)>0 or len(unsatisfied_clauses)>0)):
            clause = unsatisfied_clauses.popleft()
            
            literals = deque(clause) 
            while (len(literals) > 0):
                literal = literals.pop()
                index = self.get_index(literal)
                negation = self.is_negation(literal)
                
                if (index in watch_variables):
                    if (self.is_fully_satisfied_by_variable(literal, watch_variables[index], clause)):
                        break
                    else: 
                        continue
                          
                watch_variables[index] = not negation
                del unset_variables[index]
                    
        print(watch_variables)     
        print(unset_variables)     
        print(unsatisfiable_clauses)       
        if (len(unset_variables)>0 or len(unsatisfied_clauses)>0):
            return "Unsatisfiable"
        
        self.variables = watch_variables
        
        return watch_variables 
                
                
    def get_number_of_satisfied_clauses(self):
        
        counter_satisfied = 0
        for clause in self.clauses:
            if (self.verify_clause(clause)):
                counter_satisfied += 1
                
        return counter_satisfied        
            
        print(self.variables)


    def propose_solution(self):
        pass

           
if __name__ == "__main__":
    
    input_value, solution_path = INPUT, OUTPUT
    
    if (len(sys.argv) > 1):
        input_value = sys.argv[1]
    
    if (len(sys.argv)>2):
        solution_path = sys.argv[2]
        
    solutions = [file for file in listdir(solution_path) if isfile(join(solution_path, file))]
    
    for solution in solutions:
        satVerifier = SATVerifier(input_value, join(solution_path, solution))
        if (satVerifier.verify_solution()):
            print(solution)
            
    
    satVerifier = SATVerifier(input_value)
    
    #print(satVerifier.watch_list_heuristic())
    
    node = satVerifier.binary_tree_search()
    
    
    
    print(satVerifier.verify_solution())


        
        
    
    
    
    