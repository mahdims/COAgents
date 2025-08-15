# The code for performing parallel beam search
import typing
import os, sys, time

import multiprocessing as mp
from multiprocessing import shared_memory
from functools import partial

import numpy as np

# This routine converts vrp routes representation from a numpy array into list of routes
def list_routes(routes : np.ndarray) -> list :
    routes_list = []
    depot = routes[0]
    depot_indices = np.arange(routes.size, dtype=routes.dtype)[(routes == depot)]
    i = 0
    while i < depot_indices.size - 1:
        if depot_indices[i + 1] - depot_indices[i] > 1:
            routes_list.append(routes[depot_indices[i]: depot_indices[i + 1] + 1].tolist())
        i += 2
    return routes_list
# This routine quickly computes cost of the route but don't check for feasibilities
def cost_routes(routes : np.ndarray, times_matrix : np.ndarray) -> float :
    return times_matrix[routes[1 :], routes[: -1]].sum()


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Add the directory to sys.path
sys.path.append(os.path.join(BASE_DIR, '../heuristics'))
from InstanceReader import InstanceReader
#from ParameterReader import ParameterReader
from VRPSolution import Solution

# Compute probabilistic beam
def stable_softmax_numpy(x : np.ndarray) -> np.ndarray :
    z = x.astype(np.float64)
    z = z - np.max(z)  # Subtract max to improve numerical stability
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores)
def partial_beam_search_vrp(routes_mutable : np.ndarray, beam_width : int,
                            depot : int, nNodes : int, capacity : float,
                            times_matrix : np.ndarray, demand : np.ndarray, time_windows : np.ndarray, time_services : np.ndarray,
                            t : np.ndarray, a : np.ndarray) -> np.ndarray :
    # Create the constrained mask.
    # The mutations will involve beam search withing only routes_mutable
    routes_boundary = np.concatenate((np.zeros(shape=(1,), dtype=np.bool), ((t[:-1] == depot) & (t[1:] == depot)),), dtype=np.bool)
    routes_size = np.max(np.where(t != depot)[0]) + 2
    routes_mutable_mask = np.isin(np.cumsum(routes_boundary, dtype=t.dtype), routes_mutable)[: routes_size]
    routes_immutable_size = routes_size - routes_mutable_mask.sum()
    clients_pool = t[:routes_size][routes_mutable_mask]
    clients_pool = np.concatenate((clients_pool[clients_pool != depot], np.full(shape=(1,), fill_value=depot, dtype=t.dtype)), axis=0)
    clients_pool = np.tile(clients_pool,  (1, 1))
    t_new = np.full(shape=(routes_size - routes_mutable_mask.sum() + 3 * clients_pool.size - 3,), fill_value=depot, dtype=t.dtype)
    t_new[: routes_immutable_size] = t[: routes_size][~routes_mutable_mask]
    t_new = np.tile(t_new, (beam_width, 1))

    # The beam search
    current_time = np.zeros(shape=(1,), dtype=np.float32)
    current_demand = np.zeros(shape=(1,), dtype=np.float32)
    current_routes = np.full(shape=(1, clients_pool.shape[1] * 3 - 3), fill_value=depot, dtype=t.dtype)
    current_routes_n = np.ones(shape=(1,), dtype=t.dtype)
    num_ready = 0 ; distance_min = float('inf')
    while current_routes.shape[0] != 0 :

        # Extend each active item in a beam with all possible clients
        n = (clients_pool != depot).sum(axis=1) + 1
        row_indx = np.repeat(np.arange(n.size, dtype=n.dtype), n, axis=0)
        col_indx = np.arange(n.sum(), dtype=n.dtype) - np.repeat(np.concatenate((np.zeros(shape=(1,), dtype=n.dtype), np.cumsum(n, dtype=n.dtype)[:-1]), axis=0), n, axis=0)
        current_client = clients_pool[row_indx, col_indx]
        previous_row = np.repeat(np.arange(n.shape[0], dtype=n.dtype), n, axis=0)
        previous_col = np.repeat(current_routes_n - 1, n, axis=0)
        previous_client = current_routes[previous_row, previous_col]
        # Filter arrival_time
        time_arrivals = time_windows[current_client, 0] ; time_departs = time_windows[current_client, 1]
        current_time_new = np.repeat(current_time, n, axis=0) + times_matrix[previous_client, current_client]
        client_wait_mask = (current_time_new < time_arrivals)
        current_time_new[client_wait_mask] = time_arrivals[client_wait_mask]
        current_time_new = current_time_new + time_services[current_client]
        clients_time_feasibility_mask = (current_time_new - 1.e-8 < time_departs) & (current_time_new + times_matrix[current_client, depot] - 1.e-8 < time_windows[depot, 1])
        # Filter demand
        current_demand_new = np.repeat(current_demand, n, axis=0) + demand[current_client]
        clients_demand_feasibility_mask = (current_demand_new - 1.e-8 < capacity)
        # Filter cost
        distance = np.repeat((times_matrix[current_routes[:,: -1], current_routes[:, 1:]].sum(axis=1)), n, axis=0) + times_matrix[previous_client, current_client] + times_matrix[current_client, depot]
        clients_distance_optimality_mask = (distance < (1. + 1.e-8) * distance_min)
        # Filter feasible clients
        clients_feasibility_mask = ( ( (clients_time_feasibility_mask & clients_demand_feasibility_mask) & (~((current_client[:] == depot) & (previous_client[:] == depot))) ) & clients_distance_optimality_mask )
        if not clients_feasibility_mask.sum() : break # Is anything left to extend?
        # Select randomly
        p = a[previous_client, current_client].astype(np.float64) # Note A is non-symmetric adjacency matrix of a DAG
        p[~clients_feasibility_mask] = 0. # Zero probability of the masked clients
        p[ clients_feasibility_mask] = stable_softmax_numpy(p[clients_feasibility_mask])[:] # Normalize p so it sums to 1.
        selected_extensions = np.random.choice(np.arange(p.size, dtype=np.uint32), size=min(beam_width - num_ready, clients_feasibility_mask.sum()), replace=False, p=p)

        # Housekeeping
        # Stage 1. Update everything
        routes_new = np.full(shape=(selected_extensions.shape[0], current_routes.shape[1]), fill_value=depot, dtype=current_routes.dtype)
        routes_new[:, :] = current_routes[previous_row[selected_extensions], :]
        selected_indices_flatten = np.arange(selected_extensions.size, dtype=selected_extensions.dtype)
        routes_new[selected_indices_flatten, previous_col[selected_extensions] + 1] = current_client[selected_extensions]
        current_routes = routes_new
        current_routes_n = current_routes_n[previous_row[selected_extensions]] + 1
        clients_pool = clients_pool[row_indx[selected_extensions], :]
        selected_extensions_clients_mask = (current_client[selected_extensions] != depot) # required to filter out returns to depot
        clients_pool[selected_indices_flatten[selected_extensions_clients_mask], col_indx[selected_extensions][selected_extensions_clients_mask]] = clients_pool[selected_indices_flatten[selected_extensions_clients_mask], n[row_indx[selected_extensions]][selected_extensions_clients_mask] - 2]
        clients_pool[selected_indices_flatten[selected_extensions_clients_mask], n[row_indx[selected_extensions]][selected_extensions_clients_mask] - 2] = depot
        current_time = current_time_new[selected_extensions]
        current_demand = current_demand_new[selected_extensions]
        # Stage 2. Purging clients pool if the added item was depot
        current_demand[~selected_extensions_clients_mask] = 0.
        current_time[~selected_extensions_clients_mask] = 0.
        current_routes_n[~selected_extensions_clients_mask] += 1
        # Stage 3. Offload the finished solutions
        ready_mask = ((clients_pool != depot).sum(axis=1) == 0)
        ready_rows_num = ready_mask.sum()
        if ready_rows_num :
            distance_min = min(distance_min, (times_matrix[current_routes[ready_mask, : -1], current_routes[ready_mask, 1:]].sum(axis=1)).min())
            t_new[num_ready : num_ready + ready_rows_num, routes_immutable_size : routes_immutable_size + current_routes.shape[1]] = current_routes[ready_mask, :]
            solution_routes = list_routes(t_new[0])
            current_routes = current_routes[~ready_mask]
            current_routes_n = current_routes_n[~ready_mask]
            clients_pool = clients_pool[~ready_mask]
            current_time = current_time[~ready_mask]
            current_demand = current_demand[~ready_mask]
            num_ready += ready_rows_num

    return t_new[: num_ready, :]



# The class for probabilistic beam optimization
class constrainedBeamHH(object) :
    # The init routine
    def __init__(self, instance : InstanceReader, beam_width : int = 8) -> None :
        # Setup problem variables
        self.instance = instance
        self.solution = Solution(instance)
        self.nVehicles = instance.nVehicles
        self.capacity = instance.capacity
        self.nNodes = instance.nNodes
        self.depot = int(instance.depot)
        self.times_matrix = np.full(shape=(instance.nNodes, instance.nNodes), fill_value=float('inf'), dtype=np.float32)
        for ((n0, n1), d) in instance.getTime().items(): self.times_matrix[n0, n1] = d
        assert np.max(self.times_matrix) != float('inf'), "Times are inconsistent!"
        self.demands = np.full(shape=(instance.nNodes,), fill_value=float('inf'), dtype=np.float32)
        for (n, d) in instance.demand.items(): self.demands[n] = d
        assert np.max(self.demands) != float('inf'), "Demands are inconsistent!"
        self.time_windows = np.full(shape=(self.nNodes, 2), fill_value=float('inf'), dtype=np.float32)
        for (n, w) in instance.windows.items(): self.time_windows[n, 0] = w[0]; self.time_windows[n, 1] = w[1]
        assert np.max(self.time_windows) != float('inf'), "Windows are inconsistent!"
        self.service_times = np.full(shape=(self.nNodes,), fill_value=float('inf'), dtype=np.float32)
        for (n, s) in instance.service.items(): self.service_times[n] = s
        assert np.max(self.service_times) != float('inf'), "Service are inconsistent!"

        # Own stuff
        self.beam_width = beam_width

    # The VRP cost function
    # ToDo implement the cost function
    def compute_cost(self, t : np.ndarray) -> float :
        return 0.
    # The constrained beam search
    # Note pa is the probability matrix of edges
    # ToDo: change the code when it comes to flip probability
    def constrained_partial_beam_search_vrp(self, pa : np.ndarray, t : np.ndarray) -> typing.Tuple[ float, np.ndarray ] :
        #  Select routes to mutate using mutation probability density
        # Stage I.1. Sum probability of edge mutations into probability of DAGs mutations
        t = t[: 0 if not (t != self.depot).sum() else 2 + np.arange(t.size, dtype=t.dtype)[(t != self.depot)].max()]
        mask_dag = np.concatenate(( np.zeros(shape=(1,), dtype=np.bool), (t[:-1] == self.depot) & (t[1:] == self.depot) ), axis=0)
        dag_indx = np.empty(shape=(self.nNodes,), dtype=t.dtype)
        dag_indx[t] = np.cumsum(mask_dag, dtype=t.dtype)[:] ; dag_indx[self.depot] = mask_dag.sum()
        exclude_depot = np.arange(self.nNodes, dtype=t.dtype)[(np.arange(self.nNodes, dtype=t.dtype) != self.depot)]
        pdag = np.zeros(shape=(1 + dag_indx.max(), 1 + dag_indx.max(),), dtype=np.float64)
        # Convert probability of existance into probability of flip
        # P_flip = A_{ij} * (1. - P_{ij}) + (1 - A_{ij}) * P_{ij} = | A_{ij} - P_{ij} |
        a = np.zeros(shape=pa.shape, dtype=np.bool) ; a[t[:-1], t[1:]] = True
        np.add.at(pdag, (np.repeat(dag_indx[np.newaxis, exclude_depot], self.nNodes-1, axis=0),
                         np.repeat(dag_indx[exclude_depot, np.newaxis], self.nNodes-1, axis=1)),
                         np.abs(a[exclude_depot, :][:, exclude_depot].astype(pa.dtype) - pa[exclude_depot, :][:, exclude_depot]) )
        routes_size = np.zeros(shape=(1 + dag_indx.max(), ), dtype=np.int64)
        np.add.at(routes_size, (dag_indx[t])[t != self.depot], 1.)
        assert np.all(routes_size > 0), "routes size is inconsistent!"
        #p = stable_softmax_numpy((pdag.sum(axis=1) / routes_size))
        #p = stable_softmax_numpy((pdag.sum(axis=1) - np.diag(pdag)) / routes_size)
        p = (pdag.sum(axis=1) - np.diag(pdag)) / (routes_size * (routes_size.sum() - routes_size))
        p *= 1. / p.sum()
        routes_mutable = np.random.choice(np.arange(routes_size.size, dtype=t.dtype), size=3, replace=False, p=p)

        # Run the partial beam search on the selected routes
        # Stage II. Run beam search
        t_new = partial_beam_search_vrp(routes_mutable, self.beam_width,
                                        self.depot, self.nNodes, self.capacity,
                                        self.times_matrix, self.demands, self.time_windows, self.service_times,
                                        t, pa)
        # What if the search result in no feasible continuations
        if not  t_new.size :
            return (float('inf'), t)

        # Stage III. Otherwise compute the best cost
        distances = self.times_matrix[t_new[:, : -1], t_new[:, 1:]].sum(axis=1)
        indx_min = np.argmin(distances)

        # Return the best
        return (distances[indx_min].item(), t_new[indx_min])



# The unitttest
def main() -> bool :
    # Read instance
    DATADIR = '../dataset/cvrptw_test10K/'
    fileName = 'instance_0'
    instance = InstanceReader(os.path.join(os.path.join(BASE_DIR, DATADIR), f"{fileName}.txt"))
    instance.setDistanceType(True)
    instance.read()
    # Read solution
    BKS = float('inf') ; routes = []
    with open(os.path.join(os.path.join(BASE_DIR, DATADIR), f"{fileName}.sol"), "r") as file:
        line = file.readline()
        while line :
            if not line : break
            tokens = line.strip().split()
            if len(tokens) > 2 and tokens[0] == "Route" and tokens[1][0] == "#" and tokens[1][-1] == ':' :
                assert tokens[1][1 : -1].isdigit() and int(tokens[1][1 : -1]) == len(routes) + 1, "Inconsistency detected in BKS instance solution!"
                routes.append([instance.depot,] + [ int(token) for token in tokens[2:] if token.isdigit() ] + [instance.depot,])
            elif len(tokens) == 2 and tokens[0] == "Cost" : BKS = float(tokens[1])
            line = file.readline()
    assert BKS != float('inf') and len(routes) , "Inconsistency detected in BKS instance solution!"
    t = np.full(instance.nNodes * 3 - 3, fill_value=instance.depot, dtype=np.int32)
    routes = [item for route in routes for item in route]
    t[:len(routes)] = np.asarray(routes, dtype=np.int32)[:]
    np.random.seed(2004)
    pa = np.random.rand(instance.nNodes, instance.nNodes,).astype(np.float32) # Probabilistic adjacency matrix of the graph
    np.fill_diagonal(pa, 0.)
    # Create the beam
    beam = constrainedBeamHH(instance, beam_width=256)
    (cost_new, t_new) = beam.constrained_partial_beam_search_vrp(pa, t)
    print("The best cost is {:f}".format(cost_new))
    return True


if __name__ == "__main__" :
    result = main()
    print("The beamHH unittest {:s}uccessfully completed!".format("S" if result else "Uns"))

