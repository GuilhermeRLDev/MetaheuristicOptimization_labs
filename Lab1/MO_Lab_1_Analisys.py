# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:14:33 2021

@author: Guilherme Rossetti Lima

"""
from MO_Lab1 import NearestNeighbourInsertion
from MO_Lab1 import AlternativeInsertion
from MO_Lab1 import TestDataset
from MO_Lab1 import Random
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime
import random
import sys
from os.path import join
import regex as re


TestDataset = "TSPdata\\TSP dataset\\inst-12.tsp"

class AlgAnalyser():
    def __init__(self, filename):
        self.nn_insertion = NearestNeighbourInsertion(filename)
        self.a_insertion = AlternativeInsertion(filename)
        self.random = Random(filename)
    
    def execute_search(self, heuristic):
        start = datetime.now()
        route = heuristic.search_solution()
        end = datetime.now()
        
        duration = end - start
        
        return int(duration.total_seconds()*1000), route[1]
    
    '''Run analisys given a specific file and a number of cycles'''
    def run(self, num_cycles):
        nn_stats = np.array([])
        a_stats = np.array([])
        r_stats = np.array([])
        
        for i in range(num_cycles):
            nn_stats = np.append(nn_stats, list(self.execute_search(self.nn_insertion)), axis=0)
            a_stats = np.append(a_stats, list(self.execute_search(self.a_insertion)), axis=0)
            r_stats = np.append(r_stats, list(self.execute_search(self.random)), axis=0)
       
        nn_stats = np.resize(nn_stats, (num_cycles, 2))
        a_stats = np.resize(a_stats, (num_cycles, 2))
        r_stats = np.resize(r_stats, (num_cycles, 2))
        print(nn_stats)
        print(a_stats)
        print(r_stats)
        
        print(f"Nearest neighour average execution time: {np.average(nn_stats[:,0])}ms")
        print(f"Nearest neighour average lenght: {np.average(nn_stats[:,1])}")
        
        print(f"Alternative insertion average execution time: {np.average(a_stats[:,0])}ms")
        print(f"Alternative insertion  average lenght: {np.average(a_stats[:,1])}")
        
        print(f"Random average execution time: {np.average(r_stats[:,0])}ms")
        print(f"Random average lenght: {np.average(r_stats[:,1])}")
        
        self.plot(nn_stats[:,1], a_stats[:,1], r_stats[:,1], num_cycles, f"lenght_per_cycles{self.get_file_name()}")
        self.plot(nn_stats[:,0], a_stats[:,0], r_stats[:,0], num_cycles, f"Duration_per_cycles{self.get_file_name()}")
    
    '''
    Get mean based x,y pair to draw line
    Parameters: 
        x: cycles
        y: output we are mesuring 
    '''
    def get_mean(self, x, y):
        return [np.mean(y)]*len(x)
    
    '''
    Plot results so we can visualise it more easily
    Parameters: 
        nn_stats: nearest neighbour list of outputs
        a_stats: alterative insertion list of outputs
        r_stats: random list of outputs
        filename: path to image to be saved
    Returns: 
        Plot image and save it in the drive
    '''
    def plot(self, nn_stats, a_stats, r_stats, runs, file_name):
         fig,ax = plt.subplots()
        
         x = np.arange(0, runs)
        
         '''Plot nearest stats'''
         nearest = ax.plot(x, nn_stats, label="Nearest", marker='o')
         nearest_mean = ax.plot(x, self.get_mean(x, nn_stats), label="Nearest mean", linestyle='--')
         
         '''Plot alternative stats'''
         alternative = ax.plot(x, a_stats, label="Alternative", marker='o')
         alternative_mean = ax.plot(x, self.get_mean(x, a_stats), label="Alternative mean", linestyle='--')
         
         '''Plot random stats'''
         random = ax.plot(x, r_stats, label="Random", marker='o')
         random_mean = ax.plot(x, self.get_mean(x, r_stats), label="Random mean", linestyle='--')
        
         legend = ax.legend(loc='upper right')
         
         plt.savefig(join("analysis", file_name))
         plt.show()
        
    def get_file_name(self):
        return re.sub('[^A-Za-z0-9]+', '', str(datetime.now()))+ ".png"
    
if __name__ == "__main__":
    
    filename, num_cycles =  TestDataset, 50
    
    num_params = len(sys.argv)-1
   
    if (num_params >= 1):
       filename = sys.argv[1]
       
    if (num_params >= 2):
       num_cycles = int(sys.argv[2])
    

    algAnalyser = AlgAnalyser(filename)
    algAnalyser.run(num_cycles)
    
        
        
        
        