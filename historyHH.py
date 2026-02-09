import os, sys
import itertools
import typing

import torch as th
import numpy as np

from beamHH import list_routes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, '../model'))
from datastructures import Sample, VRP_MAX_NODES, VRP_MAX_DUMMY_DEPOTS, PARAM_NEDGES_TYPE, PARAM_UI_DIM, PARAM_UP_DIM, PARAM_VI_DIM, PARAM_VP_DIM
from layerGraphEmbedding import get_rw_landing_probs

sys.path.append(os.path.join(BASE_DIR, '../heuristics'))
from InstanceReader import InstanceReader
from VRPSolution import Solution

from layerCyclicFeatureEmbedding import CyclicEmbeddingLayer
CyclicEmbedder = CyclicEmbeddingLayer(node_dim=7,
                                      embedding_dim=64,
                                      num_nodes=VRP_MAX_NODES,
                                      num_dummy_depots=VRP_MAX_DUMMY_DEPOTS)


# The routine to embed a new solution
# Note. The emebdder is not complete, some u cells (and all u-positionals) should computed withing a graph context (here just zeroed)
def embed_sample(solution : Solution, problem_type : str) -> Sample :
    assert PARAM_UI_DIM >= 8, "Too small amount of input embeddigs for the entire solution level"
    assert PARAM_VI_DIM >= 6, "Too small amount of input embeddigs for the local solution level"
    l_dim = solution.param.nNodes - 1 + VRP_MAX_DUMMY_DEPOTS

    # Construct u inputs
    u = np.zeros(shape=(1, PARAM_UI_DIM + PARAM_UP_DIM), dtype=np.float32)  # Global features
    u[0, 0] = float(solution.getTotalCost())
    u[0, 1] = float(len(solution.routes)) #solution.param.nVehicles)
    u[0, 2] = float(solution.param.nNodes - 1)
    u[0, 3] = float(solution.param.capacity) / float(sum(solution.param.demand.values()))
    u[0, 4] = 0.  # This gap should be determined in a run over the edges float(psg.nodes[node_key]['gap'])
    u[0, 5] = 0.  # The sum of neighboring solutions cost
    u[0, 6] = 0.  # The square of sum of neighboring solutions cost
    u[0, 7] = 1.  # The amount of neighboring solutions (plus self-edge)
    u[0, PARAM_UI_DIM :]   = float(1.) # Positional emebdding of a single node

    # Construct V inputs
    # ToDo : save the constant V embedding
    v = np.zeros(shape=(l_dim, PARAM_VI_DIM + PARAM_VP_DIM), dtype=np.float32)  # input values of the VRP instance
    # Features of V items
    v[: VRP_MAX_DUMMY_DEPOTS, 0] = solution.param.positions[solution.param.depot][0]
    v[: VRP_MAX_DUMMY_DEPOTS, 1] = solution.param.positions[solution.param.depot][1]
    v[: VRP_MAX_DUMMY_DEPOTS, 2] = solution.param.demand[solution.param.depot] / float(solution.param.capacity)
    if   problem_type == 'vrptw' :
        v[: VRP_MAX_DUMMY_DEPOTS, 3] = solution.param.windows[solution.param.depot][0] # VRPTW
        v[: VRP_MAX_DUMMY_DEPOTS, 4] = solution.param.windows[solution.param.depot][1] # VRPTW
        v[: VRP_MAX_DUMMY_DEPOTS, 5] = solution.param.service[solution.param.depot]    # VRPTW
    elif problem_type == 'cvrp'  :
        v[: VRP_MAX_DUMMY_DEPOTS, 3] = 0. # CVRP
        v[: VRP_MAX_DUMMY_DEPOTS, 4] = 0. # CVRP
        v[: VRP_MAX_DUMMY_DEPOTS, 5] = 0. # CVRP
    else                         :
        raise NotImplementedError(f"Type of problems {problem_type} is not supported!")
    v[:, 7 : PARAM_VI_DIM] = float(0.)
    for indx in solution.param.positions.keys():
        assert indx < solution.param.nNodes, "Error in position information of VRP instance!"
        if indx == solution.param.depot : continue
        i = VRP_MAX_DUMMY_DEPOTS + indx - int(indx > solution.param.depot)
        (v[i, 0], v[i, 1]) = solution.param.positions[indx]
        v[i, 2]            = solution.param.demand[indx] / float(solution.param.capacity)
        if   problem_type == 'vrptw' :
            (v[i, 3], v[i, 4]) = solution.param.windows[indx] # CVRPTW
            v[i, 5]            = solution.param.service[indx] # CVRPTW
        elif problem_type == 'cvrp'  :
            (v[i, 3], v[i, 4]) = (0., 0.) # CVRP
            v[i, 5]            = 0.       # CVRP
        else                         :
            raise NotImplementedError(f"Type of problems {problem_type} is not supported!")
    # Construct the vertices order
    s = np.zeros(shape=(1, solution.param.nNodes - 1, 2), dtype=np.uint32)
    size_s = 0 ; size_d = 1
    for route in solution.routes :
        if len(route) <= 2 : continue
        assert route[0] == solution.param.depot or route[-1] == solution.param.depot, "Wrong route detected!"
        for indx in route :
            assert indx < solution.param.nNodes, "Error in vrp routes decoding"
            if indx == solution.param.depot :
                if size_s > 0 and s[0, size_s - 1, 1] == 0:
                    s[0, size_s - 1, 1] = size_d
                    size_d += 1
            else                            :
                s[0, size_s, 0] = indx - (1 if solution.param.depot < indx else 0)
                size_s += 1
    if size_s != solution.param.nNodes - 1 :
        print("here!")
    assert size_s == solution.param.nNodes - 1, "Error in vrp solutions table"
    # Construct positional embedding of the vrp nodes
    PFE = CyclicEmbedder.forward_pe(th.from_numpy(s.reshape(-1, 2)).to(dtype=th.int64), th.LongTensor([l_dim]), th.LongTensor([solution.param.nNodes - 1]))
    assert v.shape[0] == PFE.shape[0], "Positional and features emebeddings of the nodes are of different shape!"
    v[:, PARAM_VI_DIM :] = PFE.numpy()[:, :]
    v = v[VRP_MAX_DUMMY_DEPOTS-1:,:]
    l_dim = solution.param.nNodes

    # Construct solutions array
    assert len(solution.routes) <= solution.param.nVehicles , f"Too many routes for f{solution.param.nVehicles} cars"
    t = np.asarray(list(itertools.chain(*solution.routes)) + [solution.param.depot] * 2 * (solution.param.nVehicles - len(solution.routes)), dtype=np.int32)
    # Convert into a sample
    return Sample(n=np.ones( shape=(1,), dtype=np.int64),
                  m=np.zeros(shape=(1,), dtype=np.int64),
                  s=PARAM_NEDGES_TYPE,
                  l=l_dim,
                  u_dim=PARAM_UI_DIM + PARAM_UP_DIM,
                  v_dim=PARAM_VI_DIM + PARAM_VP_DIM,
                  u=u,
                  v=v,
                  e=np.zeros(shape=(0,3,), dtype=np.int64),
                  i_ptr=None,
                  n_arr=None,
                  g_dst=None,
                  m_bst=None,
                  t    = t)  # the last one is for target solution

# This routine removes side branches in the PSG
def purge_sample(sample : Sample) -> Sample :
    # Extract the path
    path_mask = np.zeros(shape=(sample.n[0],), dtype=bool)
    path = np.empty(shape=(sample.n[0],), dtype=sample.e.dtype)
    path_length = 0
    parents = np.argmin(sample.u[:, 0], keepdims=True)
    while parents.size == 1 :
        path_mask[parents[0]] = True
        path[path_length] = parents[0]
        path_length += 1
        parents = np.where(sample.e[:, 1] == parents[0])[0]
        assert parents.size <= 1 , "Two root from the same node are detected!"
        parents = sample.e[parents, 0]

    # Remap vertices and edges
    new_indices = np.cumsum(path_mask, dtype=np.int64) - 1
    u = np.empty(shape=(path_length, sample.u_dim), dtype=sample.u.dtype)
    v = np.empty(shape=(path_length, sample.l * sample.v_dim), dtype=sample.v.dtype)
    t = np.empty(shape=(path_length, sample.t.shape[1]), dtype=sample.t.dtype)
    u[new_indices[path[: path_length]], :] = sample.u[path[: path_length], :]
    v[new_indices[path[: path_length]], :] = (sample.v.reshape(sample.n[0], -1))[path[: path_length], :]
    t[new_indices[path[: path_length]], :] = sample.t[path[: path_length], :]
    e = sample.e[(path_mask[sample.e[:, 0]] & path_mask[sample.e[:, 1]]), :]
    assert e.shape[0] == path_length - 1 , "For the purged graph n_vertices == n_edges + 1"
    e[:, 0] = new_indices[e[:, 0]] ; e[:, 1] = new_indices[e[:, 1]]

    # Convert into a sample
    return Sample(n=np.asarray([path_length,], dtype=sample.n.dtype),
                  m=np.asarray([path_length - 1,], dtype=sample.m.dtype),
                  s=sample.s,
                  l=sample.l,
                  u_dim=sample.u_dim,
                  v_dim=sample.v_dim,
                  u=u,
                  v=v.reshape(-1, sample.v_dim),
                  e=e,
                  i_ptr=None,
                  n_arr=None,
                  g_dst=None,
                  m_bst=None,
                  t    = t)  # the last one is for target solution

# The history of HH class
class HistoryHH(list):

    # Class method
    @classmethod
    def _check_item(cls, sample : Sample) -> bool :
        if not isinstance(sample, Sample) :
            raise TypeError(f"Item must be a Sample instance, got {type(sample).__name__}")

    # The add-hoc routine to convert PSG3 into PSG2 samples
    @staticmethod
    def psg3to2(psg3 : Sample) -> typing.Tuple[np.ndarray, Sample] :
        # Create vertices mask
        vertices_mask = np.zeros(psg3.n.sum(), dtype=bool)
        vertices_mask[psg3.e[:, 0]] = True
        edges_mask = ((psg3.u[psg3.e[:, 0], 0] > psg3.u[psg3.e[:, 1], 0]) | (vertices_mask[psg3.e[:, 1]]))
        vertices_mask[:] = True
        vertices_mask[psg3.e[~edges_mask, 1]] = False
        vertices_map = np.arange(psg3.n.sum(), dtype=psg3.n.dtype)[vertices_mask]
        # Create purged sample
        def elegant_reduceat(mask : np.ndarray, sizes : np.ndarray) -> np.ndarray :
            # Mask zeros in the sizes array
            sizes_mask = (sizes != 0)
            valid_sizes = sizes[sizes_mask]
            # Compute sums only for non-zero segments
            sums = np.add.reduceat(mask, np.concatenate((np.zeros(shape=(1,), dtype=sizes.dtype), np.cumsum(valid_sizes)[:-1]), axis=0)) if valid_sizes.size > 0 else np.array([], dtype=sizes.dtype)
            # Create result array and fill valid sums
            new_sizes = np.zeros_like(sizes, dtype=sizes.dtype)
            new_sizes[sizes_mask] = sums[:]
            return new_sizes
        psg2 = Sample(
            n = elegant_reduceat(vertices_mask, psg3.n),
            m = elegant_reduceat(   edges_mask, psg3.m),
            l = psg3.l,
            u_dim = psg3.u_dim,
            v_dim = psg3.v_dim,
            u = psg3.u[vertices_mask, :],
            v = psg3.v.reshape(psg3.n.sum(), -1)[vertices_mask].reshape(-1, psg3.v_dim),
            e = psg3.e[edges_mask],
            i_ptr = None,
            n_arr = None,
            g_dst = None,
            m_bst = None,
            s = psg3.s,
            t = None
        )

        # Polish edges
        vertices_imap = np.cumsum(vertices_mask, dtype=psg3.n.dtype) - 1
        psg2.e[:, 0] = vertices_imap[psg2.e[:, 0]]
        psg2.e[:, 1] = vertices_imap[psg2.e[:, 1]]
        # Return the purged graph
        return (vertices_map, psg2)


    # Set instances params
    def _init_constants(self, NUMBER_NODES : int = 101, NUMBER_VEHICLES : int = 25,
                              MAX_SAMPLES : int = 32, MAX_SOLUTIONS : int = 2147483647) -> None :
        assert MAX_SAMPLES > 1 , "To perform the search amount of memorized samples should be > 1"
        self.NUMBER_NODES = NUMBER_NODES
        self.NUMBER_VEHICLES = NUMBER_VEHICLES
        self.MAX_SAMPLES = MAX_SAMPLES
        self.MAX_SOLUTIONS = MAX_SOLUTIONS


    # The Init-method
    def __init__(self, iterable : typing.Iterable = (), verbose : bool = True) -> None :
        if verbose :
            print('''
        HistoryHH Version 1.0.1
         - it keep MAX_SAMPLES local minimums and up to MAX_SOLUTIONS solutions in a local minimum
         - it updates memories so that the last graph is \'PSG3\' and all historical are purged to \'PSG2\'
         - it never forgets the global minimum
        ''')

        for sample in iterable :
            self.__class__._check_item(sample)
        super().__init__(iterable)
        # Maximal number of subgraphs in the history
        self.MAX_SAMPLES = 32
        # Maximal number of solutions in one subgraph
        self.MAX_SOLUTIONS = 2147483647 # 2**31 - 1

    # The modified methods
    def append(self, sample : Sample) -> None :
        self.__class__._check_item(sample)
        super().append(sample)
    def insert(self, index : int, sample : Sample) -> None :
        self.__class__._check_item(sample)
        super().insert(index, sample)
    def extend(self, iterable : typing.Iterable = ()) -> None :
        for sample in iterable :
            self.__class__._check_item(sample)
        super().extend(iterable)

    # History update routine
    # ToDo: limit samples history to S_MAX solutions
    # Note the indexing here is reverse
    def update(self, vertex : int, heuristic : int, sample : Sample) -> bool :
        assert sample.n.shape[0] == 1 and sample.n[0] == 1 and sample.m.shape[0] == 1 and sample.m[0] == 0 , "Sample is a subgraph, not a stand alone solution."
        assert sample.u_dim == PARAM_UI_DIM + PARAM_UP_DIM , "Unknown u-embedding of the memorizing sample."
        assert sample.v_dim == PARAM_VI_DIM + PARAM_VP_DIM , "Unknown V-embedding of the memorizing sample."
        assert len(self) == 0 or self[0].l == sample.l , "Inner dimensionality of the problem doesn't match."
        assert type(sample.t) == np.ndarray and len(sample.t.shape) == 1 and sample.t.size == self.NUMBER_NODES - 1 + 2 * self.NUMBER_VEHICLES , "Inconsistent input solution!"
        if heuristic == PARAM_NEDGES_TYPE : # The heuristic is a jump
            if len(self) : # Purge the local search graph if there is any
                self[-1] = purge_sample(self[-1])
            if len(self) > self.MAX_SAMPLES :
                del self[np.argmax(self.get_cost()) < self[0].n[0]] # We should preserve the sample with the best local minimum found so far
            if len(sample.t.shape) == 1 : sample.t = sample.t[np.newaxis, :]
            self.append(sample)
        else                              : # The heuristic is a local search step
            target_sample = len(self) - 1
            # Save a route
            self[target_sample].t = sample.t[np.newaxis, :] if self[target_sample].t is None else np.concatenate([self[target_sample].t, sample.t[np.newaxis, :]], axis=0)
            # Save the embedding
            self[target_sample].n += 1
            assert self[target_sample].n <= self.MAX_SOLUTIONS , "Error. Not an implemented feature - too many solutions in one sample"
            self[target_sample].m += 1
            assert self[target_sample].l == sample.l , "Sample l and l in the history database must doesn't match!"
            assert self[target_sample].s == sample.s , "Sample s and s in the history database must doesn't match!"
            assert self[target_sample].u_dim == sample.u_dim , "Sample u_dim and u_dim in the history database must doesn't match!"
            assert self[target_sample].v_dim == sample.v_dim , "Sample v_dim and v_dim in the history database must doesn't match!"
            self[target_sample].e = np.concatenate((self[target_sample].e, np.zeros(shape=(1, 3,), dtype=np.int32)), axis=0)
            self[target_sample].e[-1, 0] = vertex
            self[target_sample].e[-1, 1] = self[target_sample].n[0] - 1
            self[target_sample].e[-1, 2] = heuristic
            self[target_sample].u = np.concatenate((self[target_sample].u, sample.u), axis=0)
            # Now a hard part, update the local graph context
            self[target_sample].u[-1, 4] =  self[target_sample].u[self[target_sample].e[-1, 0], 0] - self[target_sample].u[self[target_sample].e[-1, 1], 0] # gap from previous solution
            self[target_sample].u[-1, 5] = 0. # self[target_sample].u[-1, 5] =  self[target_sample].u[self[target_sample].e[-1, 0], 0]       # The sum of neighboring solutions cost
            self[target_sample].u[-1, 6] = 0. # self[target_sample].u[-1, 6] = (self[target_sample].u[self[target_sample].e[-1, 1], 0] - self[target_sample].u[self[target_sample].e[-1, 0], 0] ) ** 2  # The square of sum of neighboring solutions cost
            self[target_sample].u[-1, 7] = 1. #self[target_sample].u[-1, 7] = 1.                                                            # The amount of neighboring solutions (plus self-edge)
            self[target_sample].u[self[target_sample].e[-1, 0], 5] +=  self[target_sample].u[-1, 0]      # The sum of neighboring solutions cost
            self[target_sample].u[self[target_sample].e[-1, 0], 6] += (self[target_sample].u[-1, 0] - self[target_sample].u[self[target_sample].e[-1, 0], 0])** 2  # The square of sum of neighboring solutions cost
            self[target_sample].u[self[target_sample].e[-1, 0], 7] += 1.                                 # The amount of neighboring solutions (plus self-edge)

            self[target_sample].u[:, PARAM_UI_DIM:] = get_rw_landing_probs(PARAM_UP_DIM, self[target_sample].e[:, : 2], num_nodes=self[target_sample].n[0])
            self[target_sample].v = np.concatenate((self[target_sample].v, sample.v), axis=0)

        return True
    # History sampler
    # It is similar to the sample solutions routine from graph_sampler module
    # It aggregates up to n_samples previous subgraphs
    def sample(self, n_samples : typing.Optional[ int ] = None) -> Sample :
        assert len(self), "Sampling from the empty history is not possible. Please generate at least a random solution to start with."
        n_samples = min(n_samples, len(self)) if n_samples is not None else len(self)

        n = np.concatenate([ s.n for s in self[-n_samples:] ], axis=0)
        m = np.concatenate([ s.m for s in self[-n_samples:] ], axis=0)
        s = PARAM_NEDGES_TYPE
        l = np.array([self[0].l])
        u = np.concatenate([ s.u for s in self[-n_samples:] ], axis=0)
        v = np.concatenate([ s.v for s in self[-n_samples:] ], axis=0)
        e = np.concatenate([ s.e for s in self[-n_samples:] ], axis=0)
        # Update edges accordingly to the offsets
        offsets_repeated = np.repeat(np.concatenate((np.zeros(shape=(1,), dtype=np.int32), np.cumsum(n[: -1], dtype=np.int32)), axis=0), m)
        e[:, [0, 1]] += offsets_repeated[:, None]

        sample = Sample(
            n=n,  # Total number of nodes in the sample (2 nodes per edge)
            m=m,  # Number of selected edges
            s=PARAM_NEDGES_TYPE,
            l=l,
            u_dim=PARAM_UI_DIM + PARAM_UP_DIM,
            v_dim=PARAM_VI_DIM + PARAM_VP_DIM,
            u=u,
            v=v,
            e=e,
            i_ptr=None,
            n_arr=None,
            g_dst=None,
            m_bst=None,
            t=None
        )
        return sample

    # This routine returns array of instance costs for a queried subgraph
    def get_cost(self, subgraph_id : int = 0) -> np.ndarray :
        assert len(self) > subgraph_id , "The out of range subgraph is requested!"
        return self[subgraph_id].u[:, 0]

    # This routine return the route of an instance
    def get_route(self, subgraph_id : int = 0, item_id : int = 0) -> np.ndarray :
        assert len(self) > subgraph_id ,          "The out of range subgraph is requested!"
        assert self[subgraph_id].n[0] > item_id , "The out of range item is requested!"
        return self[subgraph_id].t[item_id]

    # Inflate node
    def get_solution(self, subgraph_id : int, item_id : int, sol : Solution) -> Solution :
        # Create a solution copy for the transformation
        sol.routes = list_routes(self.get_route(subgraph_id, item_id))
        sol.updateSolution()
        sol.Id = f"{item_id}"
        return sol


# This routine is copied from hyper-heuristics class and transformed into an independent function
def applyHeuristics(heuristicInx : int, solution : Solution) -> int :
    # Local Searches
    match heuristicInx :
        case 0 :
            solution.searchShift()
        case 1 :
            solution.searchInterchangeAll()
        case 2 :
            solution.searchOpt2All()
        case 3 :
            solution.crossExchangeAll(7)
        case 4 :
            solution.search2OptInterAll()
        case 5 :
            solution.pathRelocationAll()
        case 6 :
            solution.orOptAll()
        case 7: # New pyVRP heuristics from here below
            solution.exchange10()
        case 8:
            solution.exchange20()
        case 9:
            solution.exchange30()
        case 10:
            solution.exchange11()
        case 11:
            solution.exchange21()
        case 12:
            solution.exchange31()
        case 13:
            solution.exchange22()
        case 14:
            solution.exchange32()
        case 15:
            solution.exchange33()
        case 16:
            solution.swap_tails()
        case 17:
            solution.swap_routes()
        case 18:
            solution.swap_star()
        case _ :
            raise ValueError("Unknown heuristic index") 
    return heuristicInx

# Jump simulator (copied from GeneratePSGLocal)
# The HEAVY perturbation routine
def perturbSolution(solution : Solution, seed : typing.Optional[ int ] = None) -> bool :
    # Linearize and permute samples
    sequence = np.asarray([ item for routes in solution.routes for item in routes[1 : -1] ], dtype=np.int32)
    np.random.shuffle(sequence)
    # Create slicing points
    insertions = np.arange(0, len(sequence), dtype=np.int32)
    np.random.shuffle(insertions)
    n = int(np.random.randint(1, max(1, min(solution.param.getNVehicles(), len(sequence)))))
    insertions = insertions[ : n]
    insertions[n - 1] = len(sequence)
    insertions = np.sort(insertions, axis=0, kind='quicksort', order=None, stable=None)
    # Synthetise the routes
    solution.routes = [None]*n
    offset = 0
    for i in range(n) :
        solution.routes[i] = np.pad(sequence[offset : insertions[i]], 1, 'constant', constant_values=solution.param.depot).tolist()
        offset = insertions[i]
    solution.updateSolution()
    return True


import random

# The unittest routine
if __name__ == '__main__' :
    # Parameters of the unittest
    NITERATIONS = 128
    JUMP_PROBABILITY = 0.04
    DATA_DIR = os.path.join(BASE_DIR, '../dataset/')

    # Upload VRP instance
    vrp_instance = InstanceReader(os.path.join(DATA_DIR, "./solomon/C101.txt"))
    vrp_instance.setDistanceType(False)
    result = vrp_instance.data()
    vrp_instance.computeTime()
    assert result is not None, "Failed to load vrp_instance"
    # Wrap the solution
    solution = Solution(vrp_instance)
    solution.setRandom(random.Random()) # Heuristics below requires initialization of module random

    # Iterate with the history
    MAX_SUBGRAPH = 8 # Amount of local solutions in one subghraph
    N_PREVS      = 4 # Amount of previous history subgraphs to be included into the sample
    history = HistoryHH()
    history._init_constants(MAX_SAMPLES=4)
    # Generate sequence of actions
    np.random.seed(2004)  # Set the random seed
    hheuristics = np.random.choice(np.arange(0, PARAM_NEDGES_TYPE + 1, dtype=np.int32), size=NITERATIONS,
                                   p=np.append(np.full((PARAM_NEDGES_TYPE,), (1. - JUMP_PROBABILITY) / PARAM_NEDGES_TYPE, dtype=np.float32),
                                               np.full((1,),                       JUMP_PROBABILITY,                      dtype=np.float32), axis=0))
    # Initialize updates with empty
    route = np.arange(0, solution.param.nNodes, dtype=np.int32).tolist()
    route.remove(solution.param.depot)
    solution.routes[0] = [solution.param.depot,] + route + [solution.param.depot,]
    perturbSolution(solution)
    history.update(-1, N_PREVS, PARAM_NEDGES_TYPE, embed_sample(solution))
    # Iterate over the solver
    iteration = 0
    for heuristicInx in hheuristics :
        # Get history aggregate
        sample = history.sample(N_PREVS)
        instance_id = sample.n.sum() - 1 # SIMPLIFICATION: In the unit test always branch from the apex of search graph
        assert type(sample) == Sample , "The history output should be a sample"
        # Generate a new solution
        if heuristicInx == PARAM_NEDGES_TYPE :
            # Jump simulation
            perturbSolution(solution)
        else                      :
            # Local search simulation
            applyHeuristics(heuristicInx, solution)
        # Append the new history
        history.update(instance_id, N_PREVS, heuristicInx, embed_sample(solution))
        #print("Iteration {:d} heuristics {:d} loss {:f} edges {!r}".format(iteration, heuristicInx, solution.getTotalCost(), sample.e))
        iteration += 1

    # Final message
    result = True
    print("The unittest was {:s}".format("PASSED" if result else "FAILED"))

