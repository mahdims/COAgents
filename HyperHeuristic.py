import random
import typing
import networkx as nx
from os import path
import torch as th
import math
import sys
import numpy as np
import time

PARENTDIR = path.abspath(path.join(path.dirname(__file__), '.'))
# sys.path.append(PARENTDIR)
sys.path.append(path.join(PARENTDIR , "./heuristics/"))
sys.path.append(path.join(PARENTDIR , "./model/"))
from ALNSLocal import ALNSLocal
from Algorithm import Algorithm
from ParameterReader import ParameterReader
from VRPSolution import Solution
# This have issues in parallel environment
from datastructures import Sample, PARAM_NEDGES_TYPE, PARAM_UI_DIM, PARAM_VI_DIM, PARAM_VP_DIM, PARAM_UP_DIM, VRP_MAX_DUMMY_DEPOTS
from layerGraphEmbedding import get_rw_landing_probs
from inferance import model_inference, build_load_model, build_load_E2E_model, deploy_model, E2E_model_inference, get_instance_settings, construct_routes_from_predictions,  beam_search_vrp
from historyHH import HistoryHH, embed_sample, applyHeuristics
from beamHH import list_routes, constrainedBeamHH
from distoryHH import destroy_edges, adj_matrix_to_routes, complete_solution,  route_to_adj_matrix, build_combined_score, ruin_repair

BASEDIR = path.dirname(path.dirname(path.abspath(__file__)))

class HyperHeuristic(Algorithm):
    def __init__(self, instance : ParameterReader, model : th.nn.Module, modelE2E : th.nn.Module, problem_type : str, device_id : int = 0, verbose : bool = True) -> None :
        if verbose :
            print('''
        HyperHeuristic Version 1.0.1
         - it performs search using conditional rule
         - it supports \'PSG2\' and \'PSG3\' types of graphs in local search
         - it tries a random node if model returns zeros in predictions
         - it jumps using \'simple\' or \'hellinger\' heuristics  
         - it jumps if there are not enough room for local exploration
        ''')

        self.instance = instance
        self.set_output = True
        self.timeLimit = 90.0
        self.iterLimit = int(self.timeLimit)
        self.nRemoval = 0
        self.nInsertion = 0
        self.best = Solution(instance)
        self.initiated = 0
        self.status = False
        self.bestValue = float('inf')
        self.time = 0.0
        self.output = ""
        self.iter = 0
        self.rand = random.Random()
        self.removal = {}
        self.insertion = {}
        self.hashValues = [] # this is for solution uniquness check
        self.r = 0.1
        self.BKS = -1
        self.seed = 323435
        self.lastIter2Global = -7
        ## Parameters to tune 
        self.phase1Iter = 8
        self.verbos =  0
        self.temperature = 2
        self.maxPSGsamples = 32 # This is the maximal number of samples in history
        self.maxPSGsolutions = 96 # This is the maximal number of nodes in one search graph
        self.focusePeriod  = 2
        self.alwaysAccept = True
        ## The ML related parameter
        self.device_id = device_id
        self.model     = model
        self.E2E_model = modelE2E
        self.problem_type = problem_type
        self.searchGraph = nx.DiGraph()
        self.searchGraphEmbeddings =  None
        self.nodeMoveFrequency = {}
        self.best_local = None
        self.history = HistoryHH(verbose=verbose)
        self.beam = constrainedBeamHH(instance, beam_width=256)

    def setSeed(self, seed: int):
        self.seed = seed
        self.rand.seed(seed)

    def setTimeLimit(self, timeLimit: float):
        self.timeLimit = timeLimit

    def setIterLimit(self, iterLimit: int):
        self.iterLimit = iterLimit
        
    def setBKS(self, BKS: float):
        # It only will be used for debuging not for actual algorithm
        self.BKS =  BKS
        
    def setInitalSol(self, sol):
        self.best = sol
        self.initiated = 1

    def setDelta(self, delta: float):
        pass
    
    def getApply(self):
        pass
    def getTransitions(self):
        pass
    
    def algorithmName(self) -> str:
        return "AIHyperHeuristic"
    
    def getTotalCost(self) -> float:
        if self.status:
            return round(self.best.getTotalCost() * 1E2) / 1E2
        return float('inf')

    def getParam(self) -> ParameterReader:
        return self.instance 
     
    def getOutput(self) -> str:
        return self.output

    def getRunTime(self) -> float:
        return round(self.time * 1E2) / 1E2

    def getIterations(self) -> int:
        return self.iter

    def stoppingCriteriaMet(self):
        
        return not (self.time < self.timeLimit and self.iter < self.iterLimit and self.gap(self.BKS, self.bestValue) > 0.001)
    
    
    def drawSearchGraph(self, save=True):
        # Output a png file that draw the updated search graph at each iteration. 
        fig = plt.figure()
        # pos = graphviz_layout(self.searchGraph, prog="dot")
        # nx.draw(self.searchGraph,pos, ax=fig.add_subplot())
        pos = nx.kamada_kawai_layout(self.searchGraph)
        # pos = nx.spectral_layout(self.searchGraph)
        nx.draw(self.searchGraph, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500, font_size=10)

        if save: 
            # Save plot to file
            matplotlib.use("Agg") 
            fig.savefig("searchGraph.png")
        else:
            # Display interactive viewer
            plt.show()
    
    def initialSG(self, current: Solution):
        self.searchGraph = nx.DiGraph()
        current.Id = "0" 
        self.searchGraph.add_node("0", sol= current.routesToString(), gap=0)
    
    def updateSG(self, current: Solution, heuristicInx: int, tmp: Solution):
        
        Id = f"{len(self.searchGraph.nodes())}"
        tmp.Id = Id
        self.searchGraph.add_node(tmp.Id, sol= tmp.routesToString(), gap=0)
        if current is not None:
            self.searchGraph.add_edge(current.Id, tmp.Id, heuristic=heuristicInx)
       # if self.searchGraph.number_of_nodes() > self.maxPSGnodes:              # commeted for now until the max_solution is add to history
       #     self.pruneSearchGraph(self.maxPSGnodes)
        # self.drawSearchGraph()
        
        
    def sampleSG(self):
        pass
    
    
    def pruneSearchGraph(self, maxNode):
        n2remove =  max(self.searchGraph.number_of_nodes() - maxNode, 0)
        removedN = 0
        sortedNodes = sorted(self.searchGraph.nodes(), key= lambda x : int(x))
        for i in  sortedNodes:
            if (removedN >= n2remove):
                break
            self.searchGraph.remove_node(i)
            removedN += 1
        
        if removedN > 0:
            # Create a mapping function
            mapping = {node: str(int(node) - n2remove) for node in self.searchGraph.nodes()}
            # Relabel the nodes
            self.searchGraph = nx.relabel_nodes(self.searchGraph, mapping)
        

    def prepareSample(self):
        return [self.searchGraphEmbeddings, ]
    
    def preparePerturbSample(self):
        return [self.searchGraphEmbeddings, ]
    
    def decodeBasedOnEdgePrediction(self, predictions):
        '''
        This function only use edge predictions to select the node and edge 
        '''
        ## categorize based on the nodes
        predictions = [predictions[i:i + self.test_data[0].s] for i in range(0, len(predictions), self.test_data[0].s)]
                
        # select the node first
        ## normalize for distance 
        smoothingCoe =  lambda x: math.exp(-0.1 * ((len(predictions) - x )-20)) # (n_inx + 1)/len(predictions) lambda x: 1 #
        
        ## normalize for exploration frequency 
        for i in range(len(predictions)):
            for m in range(len(predictions[i])):
                if (i,m) in self.nodeMoveFrequency:
                    predictions[i][m] = predictions[i][m] / math.exp(self.nodeMoveFrequency[(i,m)]) #TODO is the measure good?
        
        # Selcet the node
        nodeweights  = [sum(a) * smoothingCoe(n_inx)  for n_inx, a in enumerate(predictions)]
        self.selecedNode = self.rand.choices(list(range(len(nodeweights))), weights=nodeweights, k=1)[0]
        # Select the move 
        movesP = predictions[self.selecedNode]
        self.selectedMove = self.rand.choices(list(range(len(movesP))), weights=movesP, k=1)[0]
        
    def decodeBasedOnBothPrediction(self, nodePredictions, edgePredictions):
        
        nHeuristics = self.test_data[0].s
        edgePredictions = [edgePredictions[i:i + nHeuristics] 
                           for i in range(0, len(edgePredictions), nHeuristics)]
        
        ## normalize for exploration frequency for edge-node selection
        for i in range(len(edgePredictions)):
            for m in range(len(edgePredictions[i])):
                if (i,m) in self.nodeMoveFrequency:
                    edgePredictions[i][m] = edgePredictions[i][m] / math.exp(self.nodeMoveFrequency[(i,m)]) #TODO is the measure good?
        
        ## normalize for distance for node selection
        smoothingCoe =  lambda x: math.exp(-0.1 * ((len(nodePredictions) - x )-20)) # (n_inx + 1)/len(predictions) lambda x: 1 #
        # Selcet the node
        nodeweights  = [a * smoothingCoe(n_inx)  for n_inx, a in enumerate(nodePredictions)]
        
        for i in range(len(nodePredictions)):
            if np.sum(edgePredictions[i])==0:
                nodeweights[i]=0.0
        self.selecedNode = self.rand.choices(list(range(len(nodeweights))), weights=nodeweights, k=1)[0]
        # Select the move 
        movesP = edgePredictions[self.selecedNode]
        self.selectedMove = self.rand.choices(list(range(len(movesP))), weights=movesP, k=1)[0]

    def simpleDecoding(self, nodePredictions, edgePredictions):
        t = nodePredictions.copy()
        nHeuristics = self.test_data[0].s
        edgePredictions = [edgePredictions[i:i + nHeuristics] 
                           for i in range(0, len(edgePredictions), nHeuristics)]
        
        ## normalize for exploration frequency for edge-node selection
        for i in range(len(edgePredictions)):
            for m in range(len(edgePredictions[i])):
                if (i,m) in self.nodeMoveFrequency:
                    #edgePredictions[i][m] = edgePredictions[i][m] / math.exp(self.nodeMoveFrequency[(i,m)]) #TODO is the measure good?
                    edgePredictions[i][m] = 0 if  self.nodeMoveFrequency[(i,m)]>=1 else edgePredictions[i][m] #TODO is the measure good?

        for i in range(len(nodePredictions)):
            if np.sum(edgePredictions[i])==0:
                nodePredictions[i]=0.0
        if np.all(nodePredictions == 0) or self.iter <= self.focusePeriod + self.lastIter2Global:
           self.selecedNode = len(nodePredictions) -1
        else: 
            self.selecedNode = self.rand.choices(list(range(len(nodePredictions))), weights=nodePredictions, k=1)[0]

        # Select the move 
        movesP = edgePredictions[self.selecedNode]
        if np.all(movesP == 0):
            #print("all moves have 0 probs")
            self.selectedMove = self.rand.choices(list(range(len(movesP))), k=1)[0]
        else:
            self.selectedMove = self.rand.choices(list(range(len(movesP))), weights=movesP, k=1)[0]

    def greedyDecoding(self, predictions):
        nHeuristics = self.test_data[0].s
        predictions = [predictions[i:i + nHeuristics] 
                           for i in range(0, len(predictions), nHeuristics)]
        
        bestMoveValue = 0
        for n_inx, n_values in enumerate(predictions):            
            move_inx , max_value = max(enumerate(n_values), key=lambda pair: pair[1])
            if max_value > bestMoveValue:
                bestMoveValue = max_value
                bestMove = move_inx
                bestNode = n_inx
            
        self.selecedNode = bestNode
        self.selectedMove = bestMove
        

    # This routine predicts the most promising edge and heuristic pairs for PSG dig-in
    # Type either "PSG2" or "PSG3"
    def predictMove(self, type : str = "PSG3") -> typing.Tuple[ int, int ] :
        assert type.lower() == "PSG2".lower() or type.lower() == "PSG3".lower() , "unknown predict move type, should be either PSG2 or PSG3"
        psg3 = self.history.sample()
        local_area = psg3.n[:-1].sum()

        # Perform inference
        if type.lower() == "PSG2".lower() :
            # Convert input sample from PSG3 into PSG2
            (vertices_map, psg2) = self.history.psg3to2(psg3)

            # Call inference
            predictions = model_inference([psg2,], self.model, device_ids=[self.device_id,])

            # Inflate predictions
            node_predictions = np.zeros(shape=(psg3.n.sum()), dtype=predictions[0].dtype)
            node_predictions[vertices_map] = (predictions[0][: psg2.n.sum()])[:]
            edge_predictions = np.zeros(shape=(psg3.n.sum(), psg3.s), dtype=predictions[0].dtype)
            edge_predictions[vertices_map] = (predictions[0][psg2.n.sum() :].reshape(-1, psg2.s))[:]
        else              :
            predictions = model_inference([psg3,], self.model)
            node_predictions = predictions[0][: psg3.n.sum()]
            edge_predictions = predictions[0][psg3.n.sum() :].reshape(-1, psg3.s)

        # Mask all nodes from past local searches
        node_predictions[: local_area] = 0.
        # Mask used edges
        edges = psg3.e[psg3.e[:, 0] >= local_area, :]
        edge_predictions[edges[:, 0], edges[:, 2]] = 0.
        # Mask uphill edges
        objective = self.history.get_cost(len(self.history) - 1)
        edges_mask = objective[edges[:, 0] - local_area] <= objective[edges[:, 1] - local_area]
        edge_predictions[edges[edges_mask, 1], :] = 0.

        # Conditioned probability
        conditional_preditions = node_predictions[:, np.newaxis] * edge_predictions
        (selected_node, selected_move) = np.unravel_index(np.argmax(conditional_preditions), conditional_preditions.shape)

        # Check if we can do a move at all
        if conditional_preditions[selected_node, selected_move] == 0.0 :
            # Try to rescue the situation allowing a random node selection
            node_predictions[local_area :] = 1.
            conditional_preditions = node_predictions[:, np.newaxis] * edge_predictions
            (selected_node, selected_move) = np.unravel_index(np.argmax(conditional_preditions), conditional_preditions.shape)
            # If it is not fixable try a random direction
            if conditional_preditions[selected_node, selected_move] == 0.0 :
                edge_predictions = np.random.rand(*edge_predictions.shape)
                edge_predictions[edges[:, 0], edges[:, 2]] = 0.
                edge_predictions[edges[edges_mask, 1], :] = 0.
                conditional_preditions = node_predictions[:, np.newaxis] * edge_predictions
                (selected_node, selected_move) = np.unravel_index(np.argmax(conditional_preditions), conditional_preditions.shape)
                # The second rescuing attempt - total random
                if conditional_preditions[selected_node, selected_move] == 0.0:
                    return (-1, -1, -1) # Everything failed

        return (local_area, selected_node, selected_move)


    def predictNextHeuristic(self):
        return self.selectedMove
    

    # This routine drills around the local minimum
    def predictBeam(self, t : np.ndarray) -> typing.Optional[ Solution ] :
        # Perform inference of the heat-map model
        data = self.history.sample()
        pa = E2E_model_inference(test_samples=[data,], model=self.E2E_model, device_ids=[self.device_id,])
        local_area = data.n[:-1].sum()
        sol_id = np.argmin(self.history.get_cost(len(self.history) - 1))
        pa = pa[local_area + sol_id].reshape(self.history.NUMBER_NODES, self.history.NUMBER_NODES)
        # Run the constrained beam search on the most mutated routes
        (cost, t_new) = self.beam.constrained_partial_beam_search_vrp(pa, t)
        assert (1 + t_new.size - (t_new == t_new[0]).sum() == self.history.NUMBER_NODES).item() , "Inconsistent route generated in the cnstrained beam search"
        if cost == float('inf') :
            return None
        # Wrap the best route in the solution
        self.beam.solution.routes = list_routes(t_new)
        self.beam.solution.updateSolution()
        return self.beam.solution

    def predictJump(self, type : str = 'simple') -> Solution :
        assert type.lower() == 'simple' or type.lower() == 'hellinger' , "Unknown type of jump filtration: should be either simple or hellinger"
        data = self.history.sample()
        local_area = data.n[:-1].sum()

        # Perform inference
        predictions = E2E_model_inference(test_samples=[data,], model=self.E2E_model, device_ids=[self.device_id,])

        # Construct the solution
        if type.lower() == 'simple' : # simple selector
            # Strategy 1. Jump *from* the local minimum
            sol_id = np.argmin(self.history.get_cost(len(self.history) - 1))
            prediction = predictions[local_area + sol_id].reshape(self.history.NUMBER_NODES, self.history.NUMBER_NODES)
        else                        : # Hellinger selector
            # Strategy 2. Jump *into* the most diverse proposal
            # Stage 2.1. Create matrices for all discovered minimims
            reference_solutions = np.zeros(shape=(len(self.history), self.history.NUMBER_NODES, self.history.NUMBER_NODES), dtype=np.bool)
            for i in range(len(self.history)) :
                t = self.history[i].t[np.argmin(self.history.get_cost(i)), :]
                # Mask where at least one element in the pair is not the depot
                depot = t[0]
                pairs_first = t[:-1] ; pairs_second = t[1:]
                mask = (pairs_first != depot) | (pairs_second != depot)
                reference_solutions[i, pairs_first[mask], pairs_second[mask]] = True
            reference_solutions = reference_solutions.reshape(len(self.history), -1)
            # Stage 2.2. Compute generalized cosine distances within the explored set (to debias contributions of similar solutions)
            # Note. We need to use a symmetric metric that account for normalization otherwise it is problematic to compute dissimilarities
            reference_solutions = reference_solutions.astype(np.uint8) # uint8 is for performance
            ref_norm = np.sqrt(np.sum(reference_solutions, axis=1, dtype=np.float32))
            ref_norm = np.clip(ref_norm[:, None] * ref_norm[None, :], 1e-8, None)
            cos_similarity = (reference_solutions @ reference_solutions.T) / ref_norm
            w_k  = 1. / (1.e-8 + cos_similarity.sum(axis=1))
            # Stage 2.3. Compute generalized weighted Hellinger distances
            # For each entry (i,j) in the adjacency matrices:
            #   - If A_ij == 1, compute the term 1 − sqrt(B_ij)
            #   - If A_ij == 0, compute the term 1 − sqrt(1 - B_ij)
            # Sum all these terms over all entries (i,j).
            # The generalized Hellinger distance will be the square root of the resulting sum.
            term_ones  = ((1. - np.sqrt(     predictions[:, np.newaxis, :])) * ( reference_solutions[np.newaxis, :, :])).sum(axis=-1)
            term_zeros = ((1. - np.sqrt(1. - predictions[:, np.newaxis, :])) * (~reference_solutions[np.newaxis, :, :])).sum(axis=-1)
            hellinger_distance = np.dot(np.sqrt(term_ones + term_zeros), w_k)
            # Stage 2.4. Form the most diverse solution
            prediction = predictions[np.argmax(hellinger_distance)].reshape(self.history.NUMBER_NODES, self.history.NUMBER_NODES)

        sol = Solution(self.instance)
        sol.setRandom(self.rand)
        sol.routes = construct_routes_from_predictions( prediction, self.DEMANDS, self.TIME_WINDOWS, self.SERVICE_TIMES, self.DISTANCE_MATRIX, self.VEHICULE_CAPACITY)
        sol.updateSolution()
        return sol
    
    def predictdistroy(self, type : str = 'simple') -> Solution :
        assert type.lower() == 'simple' or type.lower() == 'hellinger' , "Unknown type of jump filtration: should be either simple or hellinger"
        rate = 0.3
        if type.lower() == 'simple':
            rate = 0.3
        if type.lower() == 'hellinger':
            rate = 0.5
        data = self.history.sample()
        (vertices_map, psg2) = self.history.psg3to2(data)
        local_area = data.n[:-1].sum()

        # Perform inference
        predictions1 = E2E_model_inference(test_samples=[psg2,], model=self.E2E_model, device_ids=[self.device_id,])
        predictions = np.zeros(shape=(data.n.sum(), predictions1.shape[1]), dtype=predictions1[0].dtype)
        predictions[vertices_map] = predictions1[:]

        sol_id = np.argmin(self.history.get_cost(len(self.history) - 1))
        prediction = predictions[local_area + sol_id].reshape(self.history.NUMBER_NODES, self.history.NUMBER_NODES)
        
        sol = Solution(self.instance)
        sol.setRandom(self.rand)
        sol = self.history.get_solution(len(self.history) - 1, sol_id, sol)
        #sol.routes = ruin_repair(sol.getRoutes(), prediction, self.DEMANDS, self.TIME_WINDOWS, self.SERVICE_TIMES, self.DISTANCE_MATRIX, self.VEHICULE_CAPACITY, 10, 10)
        currAdj =  route_to_adj_matrix(sol.getRoutes(), sol.param.nNodes)
        #prediction = build_combined_score(prediction, self.DISTANCE_MATRIX, 0.01, 0.5)
        currAdj = destroy_edges(currAdj ,  prediction, rate )
        partial_routes = adj_matrix_to_routes(currAdj)
        sol.routes = complete_solution(partial_routes , prediction, self.DISTANCE_MATRIX, self.DEMANDS, self.TIME_WINDOWS, self.SERVICE_TIMES, self.VEHICULE_CAPACITY)
        sol.updateSolution()
        return sol

    
    def intialPhase(self, current, retries : int = 3):
        self.initialSG(current)
        self.history.update(-1, PARAM_NEDGES_TYPE, embed_sample(current, problem_type=self.problem_type))

        # OYa do several starts and select the best to avoid stuck at bad starting pose
        best_algorithm = None ; best_value = None # For some reason algorithm.best is preserved (OYa)
        for startid in range(retries) :
            algorithm = ALNSLocal(self.instance)
            algorithm.setSeed(self.seed ^ ((startid * 982451653) % 2**32))
            algorithm.setIterLimit(self.phase1Iter)
            algorithm.setInitalSol(current)
            algorithm.setOutput(False)
            algorithm.solve()
            if best_algorithm is None or best_value > algorithm.best.getTotalCost() :
                best_value = algorithm.best.getTotalCost()
                best_algorithm = algorithm
            del algorithm

        self.hashValues = best_algorithm.hashValues
        current.copySolution(best_algorithm.best)
        self.best = best_algorithm.best
        
        nId = 0
        for _,_, h, sol in best_algorithm.history:
            nId += 1
            heuristicIndex = h[2]
            self.searchGraph.add_node(f"{nId}", sol= list(sol.values())[0].routesToString(), gap=0)
            self.searchGraph.add_edge(f"{nId - 1}", f"{nId}", heuristic=heuristicIndex)
            tmp_sol = list(sol.values())[0]
            self.history.update(nId - 1, heuristicIndex, embed_sample(tmp_sol, problem_type=self.problem_type))
            del tmp_sol
    

    def accept(self, current, tmp):
        
        if (self.alwaysAccept):
            return True
        
        prob = math.exp(-1.0 / self.temperature * (tmp.getTotalCost() - current.getTotalCost()))
        if self.rand.random() < prob and prob!=1:
            return True
        return False
        
    # Solver routine
    def solve(self, verbose : bool = True) :
        # Get problem data
        (self.DEMANDS, self.SERVICE_TIMES, self.TIME_WINDOWS, self.DISTANCE_MATRIX, self.VEHICULE_CAPACITY, self.NUMBER_VEHICULES) = get_instance_settings(self.instance)
        self.history._init_constants(NUMBER_NODES=self.instance.nNodes, NUMBER_VEHICLES=self.instance.nVehicles,
                                     MAX_SAMPLES=self.maxPSGsamples, MAX_SOLUTIONS=self.maxPSGsolutions)

        # Print info
        self.setSeed(self.seed)
        self.best.setRandom(self.rand)
        if verbose :
            self.print(self.instance.info() + "\n\n")
            self.print(self.addLine() + "\n")
            self.print(f"running {self.algorithmName()}\n")
            self.print(self.addLine() + "\n\n")


        if self.initiated ==0: # Initial solution
            self.best.savingsMethod()
        
        current = Solution(self.instance)
        current.copySolution(self.best)
        local_best = current.getTotalCost()
        self.bestValue = local_best

        # This is deactivated for now
        self.temperature = -self.bestValue * 0.05 / 100 / math.log(0.5)
        # self.initWeights() 
        # totals, scores = self.initMapping()
        
        self.intialPhase(current)   

        if verbose :
            self.print(self.toTable("time", 12) + "|" + self.toTable("Iteration", 10) + "|" +
                       self.toTable("best", 18) + "|" + self.toTable("ID | current", 18) + "|" +
                       self.toTable("tmp", 18) + "|" + self.toTable("Heuristic", 5) +  "|" + self.toTable("Result", 20) )
            self.print("-" * 92 + "\n")

        # The main loop
        start_time = time.time()
        self.iter = 0
        no_jumps = 0
        no_update = 0
        indx_min = np.argmin(self.history.get_cost(len(self.history) - 1))
        self.bestValue = self.history.get_cost(len(self.history) - 1)[indx_min] ; self.bestRoute = self.history.get_route(len(self.history) - 1, indx_min) # ToDo: save best solution in the history
        local_minimums = [self.bestValue,]
        while not self.stoppingCriteriaMet():
            self.iter += 1
            no_update += 1

            # Heat-map agents
            if no_update >= 12 or self.history[-1].n == self.history.MAX_SOLUTIONS : # Apply heat-map
                if self.iter > 127 and (sol := self.predictBeam(self.bestRoute)) != None : # Try constrained beam
                    if verbose: print("Constrained beam search at iteration : ", self.iter, "new cost :", sol.getTotalCost())
                else               : # Try jump
                    sol = self.predictJump(type='simple' if no_jumps < 2 else 'Hellinger')
                    if verbose: print("jump at iteration : ", self.iter, "new cost :", sol.getTotalCost())
                # Store new item in the history
                self.history.update(-1, PARAM_NEDGES_TYPE, embed_sample(sol, problem_type=self.problem_type))
                local_minimums.append(self.history.get_cost(len(self.history) - 1)[0])
                if local_minimums[-1] < self.bestValue :
                    self.bestValue = local_minimums[-1] ; self.bestRoute = self.history.get_route(len(self.history) - 1, 0) # ToDo: save best solution in the history
                no_update = 0
                no_jumps += 1
            # Hyper-heuristic agent
            else                                                                      : # Apply move
                # Construct the solution
                (local_area, predictedNode, predictedMove) = self.predictMove(type="PSG2")
                if local_area == -1 and predictedNode == -1 and predictedMove == -1 :
                    # We are at the local minimum, no local steps around - force a jump
                    no_update = float('inf')
                    continue

                # Apply the move (heuristic on the selected sol)
                sol = Solution(self.instance)
                sol.setRandom(self.rand)
                sol = self.history.get_solution(len(self.history) - 1, predictedNode - local_area, sol)
                applyHeuristics(predictedMove, sol)

                # Save the new result in the history
                self.history.update(predictedNode - local_area, predictedMove, embed_sample(sol, problem_type=self.problem_type))

                # Get the cost and classify it
                costs = self.history.get_cost(len(self.history) - 1)
                (parent_cost, current_cost) = (costs[predictedNode - local_area], costs[-1])
                if current_cost < local_minimums[-1] :
                    no_update = 0
                    local_minimums[-1] = current_cost
                    if current_cost < self.bestValue :
                        # We found new global best !!!
                        self.bestValue = current_cost ; self.bestRoute = self.history.get_route(len(self.history) - 1, costs.size - 1) # ToDo: save best solution in the history
                        whatHappend = "New best!"
                    else :
                         whatHappend = "Improoved"
                else :
                    whatHappend = "Pass"

                self.time = time.time() - start_time
                if verbose :
                    self.print(self.toTable(f"{self.time:.2f}", 12) + "|" +
                               self.toTable(str(self.iter), 10) + "|" +
                               self.toTable(f"{self.bestValue:.2f}", 18) + "|" +
                               self.toTable(f"{sol.Id} | {self.history.get_cost(len(self.history) - 1)[predictedNode - local_area]:.2f}", 18) + "|" +
                               self.toTable(f"{self.history.get_cost(len(self.history) - 1)[-1]:.2f}", 18) + "|" +
                               self.toTable(str(predictedMove), 5) + "|" +
                               self.toTable(whatHappend, 20))

        self.best = Solution(self.instance)
        self.best.setRandom(self.rand)
        self.best.routes = list_routes(self.bestRoute) # ToDo: save best solution in the history
        self.best.updateSolution()
        self.status = self.best.isFeasible()
        if verbose :
            self.print("-" * 96 + "\n\n")


    def isFeasible(self) -> bool:
        return self.status

    def setOutput(self, set: bool):
        self.set = set

    def toTable(self, str_value: str, length: int) -> str:
        """
        Add spaces to input string if length is less than specified length
        """
        if len(str_value) < length:
            str_value += ' ' * (length - len(str_value))
        return str_value

    def print(self, str_value: str):
        """
        Function for printing: system output and/or output string
        """
        if self.set:
            print(str_value)
        self.output += str_value

    def addLine(self) -> str:
        """
        This is a line for the output
        """
        return '*' * 80

    def printSolution(self):
        """
        Print solution details
        """
        self.print(f"feasible solution: {self.status}")
        self.print(f"total cost: {round(self.bestValue * 1E2) / 1E2}")
        self.print(f"total time: {self.time}\n")
        if self.status:
            for route in self.best.getRoutes():
                self.print(f"{route}")

    def getBestRoutes(self) -> list[list[int]]:
        return [list(route) for route in self.best.getRoutes()]

    def getBestTimes(self) -> list[list[float]]:
        return [list(times) for times in self.best.getRouteTimes()]


    ## The outdate funtions

    def gap(self, value1, value2):
        if value1 == -1:
            return 100
        return (value2 - value1) / value1
    
    def updateProbabilities(self, totals, scores):
        for i in range(self.nRemoval):
            value = scores[f"R{i}"] / totals[f"R{i}"] if totals[f"R{i}"] > 0 else self.removal[i]
            self.removal[i] = (1 - self.r) * self.removal[i] + self.r * value
        total_removal = sum(self.removal.values())
        for i in range(self.nRemoval):
            self.removal[i] /= total_removal

        for i in range(self.nInsertion):
            value = scores[f"I{i}"] / totals[f"I{i}"] if totals[f"I{i}"] > 0 else self.insertion[i]
            self.insertion[i] = (1 - self.r) * self.insertion[i] + self.r * value
        total_insertion = sum(self.insertion.values())
        for i in range(self.nInsertion):
            self.insertion[i] /= total_insertion

    def updateScores(self, indexes: list[int], type_update: int, totals: dict[str, int], scores: dict[str, float]):
        rem = f"R{indexes[0]}"
        ins = f"I{indexes[1]}"

        totals[rem] = totals.get(rem, 0) + 1
        totals[ins] = totals.get(ins, 0) + 1

        if type_update > -1:
            scores[rem] = scores.get(rem, 0.0) + self.sigma[type_update]
            scores[ins] = scores.get(ins, 0.0) + self.sigma[type_update]

    
    def initMapping(self):
        """
        Initializes the mapping for totals and scores.
        
        Args:
            totals (dict): Dictionary to store total counts.
            scores (dict): Dictionary to store scores.
        """
        totals = {}
        scores = {}

        # Add values for removal heuristics
        for i in range(-1, self.nRemoval):
            totals[f"R{i}"] = 0
            scores[f"R{i}"] = 0.0

        # Add values for insertion heuristics
        for i in range(-1, self.nInsertion):
            totals[f"I{i}"] = 0
            scores[f"I{i}"] = 0.0
            
        return totals , scores

    def initWeights(self):
        """
        Initializes weights for removal and insertion heuristics.
        """
        # Initialize weights for removal heuristics
        self.removal = {i: 1.0 / self.nRemoval for i in range(self.nRemoval)}

        # Initialize weights for insertion heuristics
        self.insertion = {i: 1.0 / self.nInsertion for i in range(self.nInsertion)}




def clamp_outliers(values, factor=20, extra=10):

    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return arr, np.array([], dtype=int)

    min_val    = arr.min()
    threshold  = factor * min_val
    non_out    = arr[arr <= threshold]
    # if everything is an outlier, fall back to min_val
    max_non_out = non_out.max() if non_out.size > 0 else min_val

    out_idx    = np.where(arr > threshold)[0]
    arr[out_idx] = max_non_out + extra
    return arr, out_idx
