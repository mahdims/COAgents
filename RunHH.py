
import typing
from os import path
import torch as th
import sys
from HyperHeuristic import HyperHeuristic
import time

PARENTDIR = path.abspath(path.join(path.dirname(__file__), '..'))

sys.path.append(path.join(PARENTDIR , "./heuristics/"))
sys.path.append(path.join(PARENTDIR , "./model/"))
from InstanceReader import InstanceReader
from VRPSolution import Solution

BASEDIR = path.dirname(path.dirname(path.abspath(__file__)))
reverseStats = {i: {"gaps":[], "tries": []} for i in range(7)}


# Running Hyper Heuristic
def runHH(fileName : str, model : th.nn.Module, modelE2E : th.nn.Module, data_dir : str, bestKnownCost : typing.Optional[ float ] = None,
          problem_type : str = 'vrptw', device_id : int = 0, verbose=True) -> typing.Tuple[ float, float, float] :
    path2file = data_dir + f"{fileName}.txt"
    assert path.isfile(path2file) , "File {:s} is not found".format(path2file)
    # algorithm parameters
    limtime = 2000.0
    limIter = 1000
    typeDouble = True

    # configuration
    delta = 0.025

    # read instance
    if verbose : print(f"***************** solving instance {fileName} f{'vrptw' if problem_type == 'vrptw' else 'cvrp'}*****************")
    instance = InstanceReader(path2file)
    instance.setDistanceType(typeDouble)
    instance.read()
    instance.BuildPyVRPData()
    assert instance.nVehicles <=  228 , "Too many nVechicles in instance {:s}".format(fileName)
    # set algorithm
    algorithm = HyperHeuristic(instance, problem_type=problem_type, model=model, modelE2E=modelE2E, device_id=device_id, verbose=verbose)

    # configuration
    algorithm.setDelta(delta)

    # max time running
    algorithm.setTimeLimit(limtime)
    algorithm.setIterLimit(limIter)
    if bestKnownCost :
        algorithm.setBKS(bestKnownCost)
    # set algorithm and solve
    algorithm.setSeed(int(time.time()))
    algorithm.setOutput(True)
    algorithm.solve(verbose=verbose)
    # clear parameters
    instance.clear()
    bestCost = algorithm.best.getTotalCost()

    return ( bestCost, round( (bestCost - bestKnownCost) / bestKnownCost, 5 ) , algorithm.time )



import os
if __name__ == "__main__" :

    problem_type = 'vrptw'
    gpu_id = 0
    PARENTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(os.path.join(PARENTDIR, "./model/"))
    from inferance import build_load_model, build_load_E2E_model, deploy_model
    try    :
        model    = deploy_model(build_load_model(path_checkpoint = f"/{problem_type}/checkpoints/"), device_id=gpu_id)
    except :
        print("Worker{:d}. Failed to deploy moves model from {:s} at {:d} gpu".format(gpu_id, "/checkpoints/", gpu_id))
    else   :
        print("Worker{:d}. Succeed in deploy moves model from {:s} at {:d} gpu".format(gpu_id, "/checkpoints/", gpu_id))
    try    :
        modelE2E = deploy_model(build_load_E2E_model(path_checkpoint = f"/{problem_type}/checkpointsE2E/"), device_id=gpu_id)
    except :
        print("Worker{:d}. Failed to deploy jumps model from {:s} at {:d} gpu".format(gpu_id, "/checkpointsE2E/", gpu_id))
    else   :
        print("Worker{:d}. Succeed in deploy jumps model from {:s} at {:d} gpu".format(gpu_id, "/checkpointsE2E/", gpu_id))

    (algBestCost, algNVehicles, time_to_finsih) = runHH('instance_0', model, modelE2E, './dataset/MVMoE_data_5/', 22.1681, problem_type = problem_type, device_id = gpu_id, verbose = True)

    print("Successfully completed!")