import math
import sys
from InstanceReader import InstanceReader
from ALNS import AdaptiveLargeNeighborhoodSearch
from ALNSPlus import ALNSPlus
from os import path

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
    limtime = 60.0
    limIter = sys.maxsize
    text = True
    typeDouble = False
    ejec = 1
    seed = None

    # configuration
    delta = 0.025
    eta = 2.0
    alpha = -1.0
    beta = -1.0

    # read instance
    param = InstanceReader(file)
    param.setDistanceType(typeDouble)
    param.read()

    # set algorithm
    algorithm = ALNSPlus(param)

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
    # algorithm.best.saveToJson(BASEDIR + DATADIR + f'sol/')
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

    with open('gaps.csv', mode='w', newline='') as csvFile:
        fieldnames = ['name', 'bestNVehicles', 'bestCost', 'algBestCost', 'algNVehicles', 'gap']
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()

        for name, details in data.items():
            bestNVehicles = details.get('number_of_vehicles')
            bestCost = details.get('cost')
            try:
                algBestCost, algNVehicles = run(name)
                gap = calculateGap(bestCost, algBestCost)

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
    # main()
    name  =  "C101"
    algBestCost, algNVehicles = run(name)
        
        
        
