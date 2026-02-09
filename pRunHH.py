import os, sys, time, argparse
import multiprocessing as mp
mp.set_start_method("fork", force=True)
from multiprocessing import shared_memory

import typing
import numpy as np


import json
import csv


'''

      D A T A   T R A N S F E R  

'''

NFIELDS = 5 # name_id, cost, nvechicles, time, curr_time | status, none, none, none, curr_time
MaxTimeDelay = 3600 # Time since last activity

# Initialize locks (server and client) required for shared memory data exchange
def initialize_dataserver_locks(ncpus : int) -> typing.Tuple[ typing.List[ mp.Lock ], typing.List[ mp.Lock ] ] :
    server_locks = [ mp.Lock() for _ in range(ncpus + 1) ]
    client_locks = [ mp.Lock() for _ in range(ncpus) ]
    for server_lock in server_locks :
        try :
            server_lock.release()
        except ValueError :
            pass
    for client_lock in client_locks : client_lock.acquire(block=False)
    return (server_locks, client_locks)

# Creates a shared memory scores window
def shmem_create(name : str, shape : np.shape, dtype : np.dtype, prefix : str = "pRunHH_") -> typing.Tuple[ shared_memory.SharedMemory, np.ndarray ] :
    shm_name = prefix + name
    try:
        # Try to create the shared memory block
        shmem = shared_memory.SharedMemory(name=shm_name, create=True, size=np.prod(shape) * np.dtype(dtype).itemsize)
    except FileExistsError:
        # If it already exists, unlink and create a new one
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        existing_shm.close()
        existing_shm.unlink()
        shmem = shared_memory.SharedMemory(name=shm_name, create=True, size=np.prod(shape) * np.dtype(dtype).itemsize)
    shared = np.ndarray(shape=shape, dtype=dtype, buffer=shmem.buf)
    return (shmem, shared)
# Connect to an existent shared memory segment
def shmem_connect(name : str, shape : np.shape, dtype : np.dtype) -> typing.Tuple[ shared_memory.SharedMemory, np.ndarray ] :
    shmem = shared_memory.SharedMemory(name=name, create=False)
    shared = np.ndarray(shape=shape, dtype=dtype, buffer=shmem.buf)
    return (shmem, shared)
# Deletes shared memory
def shmem_del(shmem: shared_memory.SharedMemory) -> None:
    shmem.close()
    shmem.unlink()


'''

      M A I N   W O R K E R  

'''

# The main routine of workers
def main_worker(gpu_id : int, server_locks : typing.List[ mp.Lock ], client_locks : typing.List[ mp.Lock ], shm_exchange_name : str, args : argparse.Namespace) -> None :
    print("Worker{:d}. Starting...".format(gpu_id))
    # Setup CUDA
    from RunHH import runHH
    # Load and deploy models
    PARENTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(os.path.join(PARENTDIR, "./model/"))
    from inferance import build_load_model, build_load_E2E_model, deploy_model
    try    :
        model    = deploy_model(build_load_model(path_checkpoint = f"/{args.problem_type}/checkpoints/"), device_id=gpu_id)
    except :
        print("Worker{:d}. Failed to deploy moves model from {:s} at {:d} gpu".format(gpu_id, "/checkpoints/", gpu_id))
    else   :
        print("Worker{:d}. Succeed in deploy moves model from {:s} at {:d} gpu".format(gpu_id, "/checkpoints/", gpu_id))
    try    :
        modelE2E = deploy_model(build_load_E2E_model(path_checkpoint = f"/{args.problem_type}/checkpointsE2E/"), device_id=gpu_id)
    except :
        print("Worker{:d}. Failed to deploy jumps model from {:s} at {:d} gpu".format(gpu_id, "/checkpointsE2E/", gpu_id))
    else   :
        print("Worker{:d}. Succeed in deploy jumps model from {:s} at {:d} gpu".format(gpu_id, "/checkpointsE2E/", gpu_id))

    # Connect to shared memory
    (shm_exchange, shared_exchange) = shmem_connect(name=shm_exchange_name, shape=(args.mgpus + 1, NFIELDS,), dtype=np.float64)
    # Do looping
    count = 0
    while client_locks[gpu_id].acquire(block=True, timeout=(300 + MaxTimeDelay)) :
        # Read
        instance_id = round(shared_exchange[gpu_id, 0].item())
        instance_cost = shared_exchange[gpu_id, 1].item()
        if instance_id >= len(args.instance_names) : break
        # Execute
        instance_name = args.instance_names[instance_id]
        try :
            (algBestCost, algNVehicles, time_to_finsih) = runHH(instance_name, model, modelE2E, args.data_dir, instance_cost,
                                                                problem_type=args.problem_type, device_id=gpu_id, verbose=False)
        except Exception as e :
            print("Worker{:d}. Warning, failed to solve instance {:s}".format(gpu_id, instance_name))
            (algBestCost, algNVehicles, time_to_finsih) = (0., 0., 0.)
        # Wrte and unlock
        shared_exchange[gpu_id, 0] = instance_id
        shared_exchange[gpu_id, 1] = algBestCost
        shared_exchange[gpu_id, 2] = algNVehicles
        shared_exchange[gpu_id, 3] = time_to_finsih
        shared_exchange[gpu_id, 4] = time.time() - args.born_time
        #print(f"Wordker {gpu_id} sent {instance_id}")
        server_locks[gpu_id].release()
        try :
            server_locks[-1].release()
        except ValueError :
            pass
        count = count + 1

    # Exit
    shm_exchange.close()
    print("Worker{:d}. Finished its job after handling {:d} instances.".format(gpu_id, count))
    sys.exit(count)


'''

      S E R V E R  

'''

# The actual main
def main(args : argparse.Namespace) -> int :
    # Upload data before the fork
    fname_json = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_dir) , "BestObj.json")
    try :
        with open(fname_json, 'r') as file:
            data = json.load(file)
    except FileNotFoundError    :
        print("Master. Error: BestObj.json file not found.")
        return 0
    except json.JSONDecodeError :
        print("Master. Error: Failed to parse JSON file.")
        return 0
    else                        :
        print("Master. Uploaded statistic for {:d} instances from {:s}".format(len(data), fname_json))
    args.instance_names = [ name for name in data.keys() ]
    args.mgpus = min(args.mgpus, len(args.instance_names))

    # Open shared memory windows and create locks
    exchange_shape = (args.mgpus + 1, NFIELDS,) # name_id, nVechicles, cost, time | status, None, None, activity_time
    (shm_exchange, shared_exchange) = shmem_create(name="exchange", shape=exchange_shape, dtype=np.float64, prefix="pRunHH_")
    shared_exchange[:, 0] = -1.
    shared_exchange[:, 4] = 0.
    (server_locks, client_locks) = initialize_dataserver_locks(args.mgpus)

    # Take-off the workers
    p = []
    for cpu_id in range(args.mgpus) :
        p_ = mp.Process(target=main_worker, args=(cpu_id, server_locks, client_locks, shm_exchange.name, args))
        p_.start()
        p.append(p_)
    print("Master: Launched {:d} parallel worker processes.".format(args.mgpus))

    # Setup CUDA properly
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Give no access to CUDA
    os.environ['OMP_NUM_THREADS']      = '1'  # Prevent CPU thread oversubscription
    # Start the logger
    fieldnames = ['name', 'bestNVehicles', 'bestCost', 'algBestCost', 'algNVehicles', 'gap', 'time']
    # Open the file and write the header once
    with open(args.log_file, mode='w', newline='') as csvFile :
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()

    # Main loop
    ninstances = 0 ; nfinished = 0
    while nfinished < len(args.instance_names) and np.any(shared_exchange[-1, -1] - shared_exchange[: -1, -1] < MaxTimeDelay) :
        server_locks[-1].acquire(block=True)
        for i in range(args.mgpus) :
            # Check if a worker is done
            if server_locks[i].acquire(block=False) :
                # Read
                instance_id = round(shared_exchange[i, 0].item())
                instance_cost = shared_exchange[i, 1].item()
                instance_nvehicles = round(shared_exchange[i, 2].item())
                instance_time = shared_exchange[i, 3].item()
                # Write
                shared_exchange[i, 0] = ninstances
                shared_exchange[i, 1] = data[args.instance_names[ninstances]].get('cost') if ninstances < len(args.instance_names) else 0.
                client_locks[i].release()
                ninstances = ninstances + 1
                # Skip failed or initialization
                if instance_id != -1 :
                    # Optionally report
                    nfinished = nfinished + 1
                    if not nfinished  % 0xF :
                        print("Master: Inference progress is {:d}.".format(nfinished ))
                    # Append the row into the log
                    instance_name = args.instance_names[instance_id]
                    bestNVehicles = data[instance_name].get('number_of_vehicles')
                    bestCost = data[instance_name].get('cost')
                    with open(args.log_file, mode='a', newline='') as csvFile   :
                        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        writer.writerow({
                            'name'          : instance_name,
                            'bestNVehicles' : bestNVehicles,
                            'bestCost'      : bestCost,
                            'algBestCost'   : instance_cost,
                            'algNVehicles'  : instance_nvehicles,
                            'gap'           : round( (instance_cost - bestCost) / bestCost , 5 ),
                            'time'          : instance_time
                        })
        shared_exchange[-1, -1] = time.time() - args.born_time # Update activity

    # Finalize and exit
    shmem_del(shm_exchange)
    return ninstances


# The entry point
if __name__ == "__main__" :
    # Parse command line
    parser = argparse.ArgumentParser(
             prog="pRunHH parallel inference for datasets",
             description="It computes inferences using a HH-ML algorithm.",
             epilog="")
    #parser.add_argument('-d', '--data_dir',     default=f"./dataset/MVMoE_data_5/",     nargs='?', action="store",      type=str, dest="data_dir",     help="The path to the vrp instances")
    parser.add_argument('-d', '--data_dir',     default=f"./dataset/MVMoE_data/",       nargs='?', action="store",      type=str, dest="data_dir",     help="The path to the vrp instances")
    parser.add_argument('-l', '--log_file',     default=f"./MVMoE_data_07_08_2025.csv", nargs='?', action="store",      type=str, dest="log_file",     help="The path to the log file")
    parser.add_argument('-m', '--mgpus',        default=1,                              nargs='?', action="store",      type=int, dest="mgpus",        help="The amount of parallel GPUs involved")
    parser.add_argument('-t', '--problem_type', default="vrptw",        choices=["cvrp", "vrptw"], action="store",      type=str, dest="problem_type", help="The type of inference: \'cvrp\' or \'vrptw\'")
    parser.add_argument('-v', '--verbose',      default=False,                                     action="store_true",           dest="verbose",      help="Enable verbose mode")
    args = parser.parse_args()
    args.born_time = time.time()
    args.device_ids = [ gid for gid in range(args.mgpus) ]

    # pRun
    ninstances = main(args)

    print("Processed all instances from {:s} using {:d} parallel (gpu-augmented) threads in {:f}s. The result are in {:s} log file.".format(args.data_dir, args.mgpus, time.time() - args.born_time, args.log_file))

