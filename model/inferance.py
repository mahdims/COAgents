import typing
import os
from os import path
import numpy as np
import torch as th

from datastructures import ( Sample, PARAM_UI_DIM, PARAM_UP_DIM, PARAM_VI_DIM, PARAM_VP_DIM, PARAM_NEDGES_TYPE )
from model import ( PARAM_U_DIM, PARAM_V_DIM,
                    PARAM_EZ_DIM, PARAM_GGCN0_Z_DIM, PARAM_GGCN0_DROP, PARAM_TRANS0_NHEAD,
                    PARAM_TRANS0_Z_DIM, PARAM_TRANS0_DROP, PARAM_LOSS_Z_DIM, PARAM_LEARNING_RATE,
                    PARAM_DACT_NHEAD, PARAM_DACT_HIDDEN, HeuristicNet )
from layerE2E import E2ENet
import math

BASEDIR = path.abspath(path.dirname(__file__))


def remove_module(state_dict):
    if list(state_dict.keys())[0].startswith("module."):
        # print("Detected 'module.' prefix in state_dict keys. Removing prefix for compatibility.")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)  # Remove only the first occurrence
            new_state_dict[new_key] = v
        return new_state_dict
    return state_dict

def load_model_weight(model : th.nn.Module,
                      optimizer : th.optim.Optimizer,
                      scheduler : th.optim.lr_scheduler.StepLR,
                      checkpoint_dir : str = "checkpoints") -> int :
    """
    Load the latest checkpoint from the checkpoint directory.
    
    Args:
        model (th.nn.Module): The model to load the state into.
        optimizer (th.optim.Optimizer): The optimizer to load the state into.
        scheduler (th.optim.lr_scheduler.StepLR): The scheduler to load the state into.
        checkpoint_dir (str): Directory where checkpoints are stored.
    
    Returns:
        int: The epoch of the loaded checkpoint. Returns 0 if no checkpoint is found.
    """
    epoch, checkpoint_fname = 0, None
    for fname in os.listdir(BASEDIR + checkpoint_dir):
        if fname.endswith(".pth"):
            tokens = fname[:-4].split('_')
            if tokens[-1].isdigit() and epoch < int(tokens[-1]):
                epoch = int(tokens[-1])
                checkpoint_fname = fname

    if epoch > 0 and checkpoint_fname:
        checkpoint = th.load(os.path.join(BASEDIR + checkpoint_dir, checkpoint_fname), map_location=th.device('cpu'))
        model.load_state_dict(remove_module(checkpoint['model_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #print(f"Loaded checkpoint '{checkpoint_fname}' (epoch {epoch})")
    else:
        print("No checkpoint found. Starting from scratch.")
    return epoch


def model_inference(inference_batches : typing.List[ Sample ], model: th.nn.Module, device_ids : typing.List[ int ] = [0,]) -> th.LongTensor :
    predictions = []
    with th.no_grad() :

        model.eval()  # Set the model to evaluation mode

        # Loop over the batches
        i  = 0
        while i < len(inference_batches) :
            # Swicth batch to tensor form
            batch_devices = []
            i_max = min(len(inference_batches), i + len(device_ids))
            for (device_id, bid) in enumerate(range(i, i_max)) :
                device = th.device('cpu') if device_id == -1 else th.device('cuda', device_ids[device_id])
                batch_devices.append( Sample (
                    n = th.from_numpy(inference_batches[bid].n).to(device=device),
                    m = th.from_numpy(inference_batches[bid].m).to(device=device),
                    l = th.from_numpy(inference_batches[bid].l).to(device=device),
                    s = inference_batches[bid].s,
                    u_dim = inference_batches[bid].u_dim,
                    v_dim = inference_batches[bid].v_dim,
                    u = th.from_numpy(inference_batches[bid].u.copy()).to(device=device),
                    v = th.from_numpy(inference_batches[bid].v.copy()).to(device=device),
                    e = th.from_numpy(inference_batches[bid].e.copy()).to(device=device),
                    i_ptr = None,
                    n_arr = None,
                    g_dst = None,
                    m_bst = None
                ) )

            # Forward propagate
            prediction = model.forward(*batch_devices)
            del batch_devices

            # Update the loss and clean the memory
            predictions.append(prediction.cpu().detach().numpy())

            # Iterate
            i = i_max

    return predictions


def build_load_model(path_checkpoint : str) -> th.nn.Module :
    
    # Initialize model
    model = HeuristicNet(
        PARAM_UI_DIM,
        PARAM_UP_DIM,
        PARAM_U_DIM,
        PARAM_VI_DIM,
        PARAM_VP_DIM,
        PARAM_V_DIM,
        PARAM_NEDGES_TYPE,
        PARAM_EZ_DIM,
        PARAM_GGCN0_Z_DIM,
        PARAM_GGCN0_DROP,
        PARAM_TRANS0_NHEAD,
        PARAM_TRANS0_Z_DIM,
        PARAM_TRANS0_DROP,
        PARAM_LOSS_Z_DIM ,
        False,
        False,)

    # Create optimizer and scheduler (required for loading checkpoint)
    optimizer = th.optim.Adam(model.parameters(), lr=PARAM_LEARNING_RATE, weight_decay=1e-5)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

    # Load the latest checkpoint
    load_model_weight(model, optimizer, scheduler, checkpoint_dir=path_checkpoint)

    return model


def build_load_E2E_model(path_checkpoint : str) -> th.nn.Module :
    model = E2ENet(PARAM_UI_DIM,
                         PARAM_UP_DIM,
                         PARAM_U_DIM,
                         PARAM_VI_DIM,
                         PARAM_VP_DIM,
                         PARAM_V_DIM,
                         PARAM_NEDGES_TYPE,
                         PARAM_EZ_DIM,
                         PARAM_GGCN0_Z_DIM,
                         PARAM_GGCN0_DROP,
                         PARAM_TRANS0_NHEAD,
                         PARAM_TRANS0_Z_DIM,
                         PARAM_TRANS0_DROP,
                         PARAM_DACT_NHEAD,
                         PARAM_DACT_HIDDEN,
                         PARAM_LOSS_Z_DIM,
                         True,
                         True,
                         1)
    epoch, checkpoint_fname = 0, None
    for fname in os.listdir(BASEDIR + path_checkpoint):
        if fname.endswith(".pth"):
            tokens = fname[:-4].split('_')
            if tokens[-1].isdigit() and epoch < int(tokens[-1]):
                epoch = int(tokens[-1])
                checkpoint_fname = fname

    if epoch > 0 and checkpoint_fname:
        checkpoint = th.load(os.path.join(BASEDIR + path_checkpoint, checkpoint_fname), map_location=th.device('cpu'), weights_only=True)
        model.load_state_dict(remove_module(checkpoint['model_state_dict']))
    else:
        print("No checkpoint found. Starting from scratch.")
    model.eval()
    return model


# Deploy the move model at device (device_id == -1 means cpu)
def deploy_model(model : th.nn.Module, device_id : int = -1) -> th.nn.Module :
    if device_id != -1 :
        model.to(th.device('cuda', device_id))
    return model

def E2E_model_inference(test_samples : typing.List[Sample], model : th.nn.Module, device_ids : typing.List[ int ] = [-1,]) -> th.Tensor :
    """
    Perform inference on a list of test samples using the trained model.
    """

    device = th.device('cpu') if device_ids[0] == -1 else th.device('cuda', device_ids[0])
    model.eval()

    with th.no_grad():
        batch_devices = []
        for sample in test_samples:
            batch_devices.append(Sample(
                n = th.from_numpy(sample.n).to(device=device),
                m = th.from_numpy(sample.m).to(device=device),
                l = th.from_numpy(sample.l).to(device=device),
                s = sample.s, 
                u_dim = sample.u_dim,
                v_dim = sample.v_dim,
                u = th.from_numpy(sample.u.copy()).to(device=device),
                v = th.from_numpy(sample.v.copy()).to(device=device),
                e = th.from_numpy(sample.e.copy()).to(device=device),
                i_ptr = sample.i_ptr,
                n_arr = sample.n_arr,
                g_dst = None, #th.from_numpy(sample.g_dst.copy()).to(device=device),
                m_bst =  None,
                t =  None,
            ))

        # Forward pass
        outputs = model.forward(*batch_devices)
    outputs = th.sigmoid(outputs)
    return outputs.detach().cpu().numpy()


def get_instance_settings(param):
    graph_size  = param.nNodes
    N_dems = param.getDemand()
    demands = np.zeros(graph_size, dtype=float)
    for i in range(len(demands)):
        demands[i] = N_dems[i]

    N_serv = param.getService()
    service_times = np.zeros(graph_size, dtype=float)
    for i in range(len(service_times)):
        service_times[i] = N_serv[i]

    list_time_windows= [i for i in param.getTimeWindows(). values()]
    time_windows = np.zeros((graph_size, 2), dtype=float) 
    T_wind = np.array(list_time_windows, dtype=float) 

    for i in range(len(time_windows) ):
        time_windows[i] = T_wind[i]  

    d = param.getTime()
    distance_matrix = np.zeros((graph_size, graph_size), dtype=float)
    for (i, j), v in d.items():
            distance_matrix[i, j] = v
    vehicle_capacity = param.capacity
    number_vehicles = param.nVehicles
    return demands, service_times, time_windows, distance_matrix, vehicle_capacity, number_vehicles



def construct_routes_from_predictions(probs, demands, time_windows, service_times, 
                                        distance_matrix, vehicle_capacity):
    if isinstance(probs, th.Tensor) :
        probs = probs.detach().cpu().numpy()
    graph_size = probs.shape[0]
    routes = []
    visited = np.zeros(graph_size, dtype=bool)
    real_depot_idx = 0  

    # Continue until all nodes except the depot have been visited.
    while not np.all(visited[1:]):
        route = [real_depot_idx]
        remaining_capacity = vehicle_capacity
        current_time = 0.0
        current_node = real_depot_idx

        while True:
            # Create feasibility mask:
            mask = visited.copy()
            mask = mask | (demands > remaining_capacity)
            
            # Compute arrival times from the current node to every other node.
            arrival_times = current_time + distance_matrix[current_node]
            # Determine start times at each candidate
            start_times = np.maximum(arrival_times, time_windows[:, 0])
            # Check candidate time window feasibility.
            time_feasible = (start_times <= time_windows[:, 1])
            mask = mask | (~time_feasible)
            
            # Check if candidates can return to the depot on time 
            finish_times = start_times + service_times 
            # Estimated arrival time at the depot if visiting candidate i.
            candidate_depot_arrivals = finish_times + distance_matrix[:, real_depot_idx]
            # Determine if returning from candidate i is feasible (depot's time window).
            return_feasible = candidate_depot_arrivals <= time_windows[real_depot_idx, 1]
            # For nodes other than the depot, update the mask if they cannot return to the depot on time.
            non_depot = np.arange(graph_size) != real_depot_idx
            mask[non_depot] = mask[non_depot] | (~return_feasible[non_depot])
            #depot_arrival_time = current_time + distance_matrix[current_node, real_depot_idx]
            #if depot_arrival_time > time_windows[real_depot_idx, 1]:
            #    mask[real_depot_idx] = True
            
            # Mask the probabilities for all infeasible nodes.
            node_probs = probs[current_node].copy()
            node_probs[mask] = 0
            
            # If no candidate remains, try forcing a return to the depot (if feasible).
            if node_probs.sum() == 0:
                if current_time + distance_matrix[current_node, real_depot_idx] <= time_windows[real_depot_idx, 1]:
                    next_node = real_depot_idx
                else:
                   raise Exception("  infeasibility ")
            else:
                # Normalize and select the candidate with the highest probability.
                node_probs = node_probs / node_probs.sum()
                next_node = int(np.argmax(node_probs))
            
            # If the next node is the depot, close out the route.
            if next_node == real_depot_idx:
                if len(route) > 1 :
                    route.append(real_depot_idx)
                    routes.append(route)
                    break
                else              :
                    next_node = int(1 + np.argmax(node_probs[1:]))
            
            # if all non-depot nodes have been served, try to return.
            if np.all(visited[1:]):
                if current_time + distance_matrix[current_node, real_depot_idx] <= time_windows[real_depot_idx, 1]:
                    route.append(real_depot_idx)
                    routes.append(route)
                    break
                else:
                    break
            
            # Update the route and state if a candidate node is selected.
            route.append(next_node)
            visited[next_node] = True
            remaining_capacity -= demands[next_node]
            
            # Update current time with the service at the selected candidate.
            arrival_time = current_time + distance_matrix[current_node, next_node]
            start_service = max(arrival_time, time_windows[next_node, 0])
            current_time = start_service + service_times[next_node]
            current_node = next_node

    return routes


def construct_routes_from_predictions2(probs, demands, time_windows, service_times, 
                                           distance_matrix, vehicle_capacity):
    probs = probs.detach().cpu().numpy()
    graph_size = probs.shape[0]
    routes = []
    visited = np.zeros(graph_size, dtype=bool)
    real_depot_idx = 0
    
    visited[real_depot_idx] = False 
    
    while not np.all(visited[1:]):  # Exclude the depot from the check
        route = [real_depot_idx]
        remaining_capacity = vehicle_capacity
        current_time = 0.0
        current_node = real_depot_idx
        
        while True:
            mask = visited.copy()
            mask = mask | (demands > remaining_capacity)
            
            arrival_times = current_time + distance_matrix[current_node]
            start_times = np.maximum(arrival_times, time_windows[:, 0])
            time_feasible = (start_times <= time_windows[:, 1])
            mask = mask | (~time_feasible)
            
            node_probs = probs[current_node].copy()
            node_probs[mask] = 0
            
            if node_probs.sum() == 0:
                next_node = real_depot_idx
            else:
                node_probs = node_probs / node_probs.sum()
                next_node = int(np.argmax(node_probs)) 
                
            if next_node == real_depot_idx:
                if len(route) > 1:
                    route.append(real_depot_idx)
                    routes.append(route)
                break
            
            if np.all(visited[1:]):
                break
            
            # Cast next_node to int before appending
            route.append(next_node)
            visited[next_node] = True
            remaining_capacity -= demands[next_node]
            
            arrival_time = arrival_times[next_node]
            start_service = max(arrival_time, time_windows[next_node, 0])
            current_time = start_service + service_times[next_node]
            current_node = next_node
    return routes

#unique beams
def beam_search_vrp( prob_matrix, demands, time_windows, service_times, distance_matrix, vehicle_capacity, beam_width, max_vehicles=None):

    N = prob_matrix.shape[0]
    num_customers = N - 1

    init = {
        "routes": [],
        "current_route": [0],
        "visited": set(),
        "remaining_capacity": vehicle_capacity,
        "current_time": 0.0,
        "log_prob": 0.0,
        "total_distance": 0.0,
        "vehicles_used": 0
    }

    beam = [init]
    completed = []

    while beam:
        # Dictionary to hold unique states for the next beam based on canonical key
        seen_states = {}

        # Create a canonical key for a state 
        def get_canonical_key(state):
            # Each route becomes a tuple for hashability
            canonical_completed_routes = tuple(sorted([tuple(r) for r in state["routes"]]))

            # Current route (as a tuple)
            canonical_current_route = tuple(state["current_route"])

            # Visited customers (as a frozenset)
            canonical_visited = frozenset(state["visited"])

            # 4. Other 
            canonical_remaining_capacity = state["remaining_capacity"]
            canonical_current_time = state["current_time"]
            canonical_vehicles_used = state["vehicles_used"]

            # Combine all componentse
            return (
                canonical_completed_routes,
                canonical_current_route,
                canonical_visited,
                canonical_remaining_capacity,
                canonical_current_time,
                canonical_vehicles_used
            )


        for cand in beam:
            last = cand["current_route"][-1]

            # If all customers done, try to close with a depot return
            if len(cand["visited"]) == num_customers:
                if last != 0:
                    ret_time = cand["current_time"] + distance_matrix[last, 0]
                    if ret_time <= time_windows[0, 1]:
                        ret_prob = prob_matrix[last, 0]
                        log_ret_prob = math.log(ret_prob) if ret_prob > 0 else float("-inf")
                        new = {
                            **cand,
                            "routes": cand["routes"] + [cand["current_route"] + [0]],
                            "current_route": [0],
                            "remaining_capacity": vehicle_capacity,
                            "current_time": 0.0,
                            "log_prob": cand["log_prob"] + log_ret_prob,
                            "total_distance": cand["total_distance"] + distance_matrix[last, 0],
                            "vehicles_used": cand["vehicles_used"] + 1
                        }
                        completed.append(new)
                else:
                    completed.append(cand)
                continue


            # 1) Try extending to each unvisited customer j
            for j in range(1, N):
                if j in cand["visited"]:
                    continue
                if demands[j] > cand["remaining_capacity"]:
                    continue

                travel_time = distance_matrix[last, j]
                arrival = cand["current_time"] + travel_time
                arrival = max(arrival, time_windows[j, 0])
                if arrival > time_windows[j, 1]:
                    continue

                depart = arrival + service_times[j]

                if depart + distance_matrix[j, 0] > time_windows[0, 1]:
                    continue

                pij = prob_matrix[last, j]
                if pij <= 0:
                    continue

                new_cand = {
                    "routes": cand["routes"].copy(),
                    "current_route": cand["current_route"] + [j],
                    "visited": cand["visited"] | {j},
                    "remaining_capacity": cand["remaining_capacity"] - demands[j],
                    "current_time": depart,
                    "log_prob": cand["log_prob"] + math.log(pij),
                    "total_distance": cand["total_distance"] + distance_matrix[last, j],
                    "vehicles_used": cand["vehicles_used"]
                }

                # --- Duplicate Detection ---
                key = get_canonical_key(new_cand)
                if key not in seen_states:
                    seen_states[key] = new_cand
                # --- End Duplicate Detection ---


            # 2) Optionally close current route by returning to depot
            if last != 0 and (max_vehicles is None or cand["vehicles_used"] + 1 < max_vehicles):
                ret_time = cand["current_time"] + distance_matrix[last, 0]
                if ret_time <= time_windows[0, 1]:
                    ret_prob = prob_matrix[last, 0]
                    log_ret_prob = math.log(ret_prob) if ret_prob > 0 else float("-inf")

                    new_cand_closed_route = {
                        **cand,
                        "routes": cand["routes"] + [cand["current_route"] + [0]],
                        "current_route": [0],
                        "remaining_capacity": vehicle_capacity,
                        "current_time": 0.0,
                        "log_prob": cand["log_prob"] + log_ret_prob,
                        "total_distance": cand["total_distance"] + distance_matrix[last, 0],
                        "vehicles_used": cand["vehicles_used"] + 1
                    }

                    # --- Duplicate Detection ---
                    key = get_canonical_key(new_cand_closed_route)
                    if key not in seen_states:
                         seen_states[key] = new_cand_closed_route
                    # --- End Duplicate Detection ---


        # the new_beam is the list of unique states found (the values in seen_states)
        new_beam = list(seen_states.values())


        # sort by log_prob and keep top-k
        new_beam.sort(key=lambda c: c["log_prob"], reverse=True)
        beam = new_beam[:beam_width]

    # pick best completed (highest log_prob)
    if completed:
        best = max(completed, key=lambda c: c["log_prob"])
        best = min(completed, key=lambda s: s["total_distance"])
        return best["routes"]#, completed 
    else:
        raise RuntimeError("No feasible solution found.")
