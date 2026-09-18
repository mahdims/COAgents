
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
    nVehicles = len(algorithm.best.getRoutes())

    # Returns (cost, number of vehicles, runtime). Callers compute the gap to the BKS themselves.
    return ( bestCost, nVehicles, algorithm.time )



import os, json, argparse
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(
             prog="RunHH single-instance solver",
             description="Solve one VRP instance with COAgents and print a verbose search trace.")
    parser.add_argument('-d', '--data_dir',     default="./dataset/MVMoE_data/", type=str, dest="data_dir",     help="Directory holding instance_*.txt and BestObj.json")
    parser.add_argument('-i', '--instance',     default="instance_0",            type=str, dest="instance",     help="Instance name without the .txt extension")
    parser.add_argument('-t', '--problem_type', default="vrptw", choices=["cvrp", "vrptw"], type=str, dest="problem_type", help="'cvrp' or 'vrptw'")
    parser.add_argument('-g', '--gpu',          default=0,                       type=int, dest="gpu_id",       help="CUDA device id (-1 for CPU)")
    args = parser.parse_args()

    problem_type = args.problem_type
    gpu_id = args.gpu_id
    data_dir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'

    # Best-known cost is optional; it is used for the gap print-out and the early-stopping rule.
    bestKnownCost = None
    try :
        with open(os.path.join(data_dir, "BestObj.json"), 'r') as f :
            bestKnownCost = json.load(f)[args.instance].get('cost')
    except (FileNotFoundError, KeyError) :
        print("Warning: no best-known cost found for {:s} in {:s}".format(args.instance, data_dir))

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

    (algBestCost, algNVehicles, time_to_finsih) = runHH(args.instance, model, modelE2E, data_dir, bestKnownCost,
                                                        problem_type = problem_type, device_id = gpu_id, verbose = True)

    if bestKnownCost :
        print("cost {:.4f} | vehicles {:d} | gap {:.2%} | time {:.1f}s".format(algBestCost, algNVehicles, (algBestCost - bestKnownCost) / bestKnownCost, time_to_finsih))
    else :
        print("cost {:.4f} | vehicles {:d} | time {:.1f}s".format(algBestCost, algNVehicles, time_to_finsih))
    print("Successfully completed!")