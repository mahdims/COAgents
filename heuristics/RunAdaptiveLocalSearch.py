import math
import sys
from InstanceReader import InstanceReader
from ALNS import AdaptiveLargeNeighborhoodSearch
from ALNSLocal import ALNSLocal
from os import path
import time

BASEDIR = path.dirname(path.abspath(__file__))
DATADIR = "/../dataset/solomon/"

import json
import csv

def run(instance, initialSol = None):
    # instance parameters
    dataConn = [None] * 3
    file = BASEDIR + DATADIR + f"{instance}.txt"

    # algorithm parameters
    type = 0
    limtime = 30.0
    limIter = 500
    text = True
    typeDouble = False
    ejec = 1
    seed = int(time.time()) #323435 # Same as the hyper heuristic

    # configuration
    delta = 0.025
    eta = 2.0
    alpha = -1.0
    beta = -1.0

    # read instance
    param = InstanceReader(file)
    param.setDistanceType(typeDouble)
    param.read()
    param.BuildPyVRPData()

    # set algorithm
    algorithm = ALNSLocal(param)

    # configuration
    algorithm.setDelta(delta)

    # max time running
    algorithm.setTimeLimit(limtime)
    algorithm.setIterLimit(limIter)
    if initialSol:
        algorithm.setInitalSol(initialSol)

    # set algorithm and solve
    if seed is None:
        seed = (type * 10000 + int(math.floor(delta * 1000)) + int(math.floor(eta * 100)) + ejec + hash(instance))
    algorithm.setSeed(seed)
    algorithm.setOutput(text)
    algorithm.solve()
    # algorithm.best.writeToJson(BASEDIR + DATADIR + 'sol/')
    # clear parameters
    param.clear()
    
    return algorithm.getTotalCost(), len(algorithm.getBestRoutes())


def calculateGap(bestCost, algBestCost):
    return (algBestCost - bestCost) / bestCost if bestCost > 0 else None


def main():
    try:
        with open(BASEDIR + DATADIR + 'BestObj.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: BestObj.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON file.")
        return
    
    outputName = 'ALNSOnlyLocal_results.csv'
    fieldnames = ['name', 'bestNVehicles', 'bestCost', 'algBestCost', 'algNVehicles', 'gap']
    # Open the file and write the header once
    with open(outputName, mode='w', newline='') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()
    
    for name, details in data.items():
        bestNVehicles = details.get('number_of_vehicles')
        bestCost = details.get('cost')
        try:
            for _ in range(5):
                algBestCost, algNVehicles = run(name)
                gap = calculateGap(bestCost, algBestCost)
                
                # Open the file in append mode to write a single row
                with open(outputName, mode='a', newline='') as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writerow({
                        'name': name,
                        'bestNVehicles': bestNVehicles,
                        'bestCost': bestCost,
                        'algBestCost': algBestCost,
                        'algNVehicles': algNVehicles,
                        'gap': gap
                    })
        except Exception as e:
            print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()
    
    # name  =  "C101"
    # algBestCost, algNVehicles = run(name)
    # print(f"{name}: Best Cost with HH {algBestCost}")
        
        
        
