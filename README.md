# MetaheuristicOptimization_labs
Labs for MSc in metaheuristic optimization

LAB-1: Notes

On the lab-1 folder you will find three files.
MO_Lab1.py: Contains implementations for nearest neighbour heuristic, alternative insertion and random insertion. You can pass the location for the tsp file when running the file and the name of the output you expect as shown bellow:

python MO_Lab1.py "{TspLocation}" "{OutputFile}" {opt}

opt: 0 = Nearest, 1 = Alternative, 2 = Random

All responses are written within the output folder.

MO_Lab1_Tests.py: Contain some unit tests for each heuristic class 

MO_Lab1_Analytics.py: Runs analysis for each heutistic, takes two parametes {0}: tsp file name, {1}: number of cycles. In order to measure performance the same heuristic multiple times and check executioin time and lenght for responses returned. This will also plot some charts in the analytics folder.

python MO_Lab1_Analytics.py "{TspLocation}" {num-of_runs}

All analytics are plotted within the analytics folder.









