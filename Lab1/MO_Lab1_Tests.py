# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:14:33 2021

@author: Guilehrme Rossetti Lima
"""
import unittest
from MO_Lab1 import NearestNeighbourInsertion
from MO_Lab1 import AlternativeInsertion
from MO_Lab1 import TestDataset
from MO_Lab1 import Random


'''Unit tests for heuristics'''
class NearestNeighbourTestHeuristics(unittest.TestCase):

    '''Tests is distance is correct'''
    def test_is_distance_correct(self):
        
        heuristic = NearestNeighbourInsertion(TestDataset)
        route = heuristic.search_solution()
        length = route[1]
        calculated_length = 0
        for i in range(len(route[0])-1): 
            calculated_length += heuristic.get_weight(heuristic.dictionary[route[0][i]], heuristic.dictionary[route[0][i+1]])
        
        self.assertEqual(calculated_length, length) 
         
    '''Tests if starts and ends at the same point'''
    def test_is_starting_ending_at_same_location(self): 
        heuristic = NearestNeighbourInsertion(TestDataset)
        route = heuristic.search_solution()
        
        self.assertEqual(route[0][0], route[0][-1]) 
    
    '''Tests if there are no duplicates'''
    def test_has_only_unique_nodes(self):
        heuristic = NearestNeighbourInsertion(TestDataset)
        route = heuristic.search_solution()
        
        '''Delete last node as it is an expected duplicatesd '''
        route_clone = route[0].copy()
        del route_clone[-1]
        has_duplicate = any(route_clone.count(element) > 1 for element in route_clone)
        
        self.assertFalse(has_duplicate)


class AlternativeTestHeuristics(unittest.TestCase):
    '''Tests is distance is correct'''
    def test_is_distance_correct(self):
        
        heuristic = AlternativeInsertion(TestDataset)
        route = heuristic.search_solution()
        length = route[1]
        calculated_length = 0
        for i in range(len(route[0])-1): 
            calculated_length += heuristic.get_weight(heuristic.dictionary[route[0][i]], heuristic.dictionary[route[0][i+1]])
        
        self.assertEqual(calculated_length, length) 
         
    '''Tests if starts and ends at the same point'''
    def test_is_starting_ending_at_same_location(self): 
        heuristic = AlternativeInsertion(TestDataset)
        route = heuristic.search_solution()
        
        self.assertEqual(route[0][0], route[0][-1]) 
    
    '''Tests if there are no duplicates'''
    def test_has_only_unique_nodes(self):
        heuristic = AlternativeInsertion(TestDataset)
        route = heuristic.search_solution()
        
        '''Delete last node as it is an expected duplicatesd '''
        route_clone = route[0].copy()
        del route_clone[-1]
        has_duplicate = any(route_clone.count(element) > 1 for element in route_clone)
        
        self.assertFalse(has_duplicate)

class RandomTestHeuristics(unittest.TestCase):
    '''Tests is distance is correct'''
    def test_is_distance_correct(self):
        heuristic = Random(TestDataset)
        route = heuristic.search_solution()
        length = route[1]
        calculated_length = 0
        for i in range(len(route[0])-1): 
            calculated_length += heuristic.get_weight(heuristic.dictionary[route[0][i]], heuristic.dictionary[route[0][i+1]])
        
        self.assertEqual(calculated_length, length) 
         
    '''Tests if starts and ends at the same point'''
    def test_is_starting_ending_at_same_location(self): 
        heuristic = Random(TestDataset)
        route = heuristic.search_solution()
        
        self.assertEqual(route[0][0], route[0][-1]) 
    
    '''Tests if there are no duplicates'''
    def test_has_only_unique_nodes(self):
        heuristic = Random(TestDataset)
        route = heuristic.search_solution()
        
        '''Delete last node as it is an expected duplicatesd '''
        route_clone = route[0].copy()
        del route_clone[-1]
        has_duplicate = any(route_clone.count(element) > 1 for element in route_clone)
        
        self.assertFalse(has_duplicate)

'''-----------------------------------------------------------------------------------------------------------------------------------'''

if __name__ == "__main__":
   unittest.main()
   
