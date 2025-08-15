from collections import defaultdict
import random
from typing import Dict, Set, Tuple
from ALNS import AdaptiveLargeNeighborhoodSearch
from ParameterReader import ParameterReader


class ALNSPlus(AdaptiveLargeNeighborhoodSearch):
    
    def __init__(self, param: ParameterReader):
        super().__init__(param)
        self.improvement: Dict[int, float] = {}
        self.nImprove = 7
    
    def updateProbabilities(self, totals: Dict[str, int], scores: Dict[str, float]):
        # Update removal probabilities
        total1 = 0.0
        for i in range(self.nRemoval):
            value = self.removal[i]
            if totals.get(f"R{i}", 0) > 0:
                value = scores[f"R{i}"] / totals[f"R{i}"]
            weight = (1 - self.r) * self.removal[i] + self.r * value
            self.removal[i] = weight
            total1 += weight
        for i in range(self.nRemoval):
            self.removal[i] /= total1

        # Update insertion probabilities
        total2 = 0.0
        for i in range(self.nInsertion):
            value = self.insertion[i]
            if totals.get(f"I{i}", 0) > 0:
                value = scores[f"I{i}"] / totals[f"I{i}"]
            weight = (1 - self.r) * self.insertion[i] + self.r * value
            self.insertion[i] = weight
            total2 += weight
        for i in range(self.nInsertion):
            self.insertion[i] /= total2

        # Update improvement probabilities
        total3 = 0.0
        for i in range(self.nImprove):
            value = self.improvement[i]
            if totals.get(f"P{i}", 0) > 0:
                value = scores[f"P{i}"] / totals[f"P{i}"]
            weight = (1 - self.r) * self.improvement[i] + self.r * value
            self.improvement[i] = weight
            total3 += weight
        for i in range(self.nImprove):
            self.improvement[i] /= total3

    def updateScores(self, indexes: Tuple[int, int, int], type: int, totals: Dict[str, int], scores: Dict[str, float]):
        # Mapping names
        rem = f"R{indexes[0]}"
        ins = f"I{indexes[1]}"
        imp = f"P{indexes[2]}"

        # Update totals
        totals[rem] += 1
        totals[ins] += 1
        totals[imp] += 1

        # Update scores
        if type > -1:
            scores[rem] += self.sigma[type]
            scores[ins] += self.sigma[type]
            scores[imp] += self.sigma[type]

    def initMapping(self):
        totals ={}
        scores ={}

        # Add values
        for i in range(-1, self.nRemoval):
            totals[f"R{i}"] = 0
            scores[f"R{i}"] = 0.0
        for i in range(-1, self.nInsertion):
            totals[f"I{i}"] = 0
            scores[f"I{i}"] = 0.0
        for i in range(-1, self.nImprove):
            totals[f"P{i}"] = 0
            scores[f"P{i}"] = 0.0
        return totals, scores

    def initWeights(self):
        # Initialize weights
        self.removal = {}
        self.insertion = {}
        self.improvement = {}

        # Initial values for removal
        for i in range(self.nRemoval):
            self.removal[i] = 1.0 / self.nRemoval

        # Initial values for insertion
        for i in range(self.nInsertion):
            self.insertion[i] = 1.0 / self.nInsertion

        # Initial values for improvement
        for i in range(self.nImprove):
            self.improvement[i] = 1.0 / self.nImprove

    def applyHeuristics(self, current) -> Tuple[int, int, int]:
        # Indexes
        indexes = [-1, -1, -1]

        # Find removal
        unif1 = random.random()
        ac1 = 0
        for i in range(self.nRemoval):
            ac1 += self.removal[i]
            if unif1 < ac1:
                indexes[0] = i
                break

        # Find insertion
        unif2 = random.random()
        ac2 = 0
        for i in range(self.nInsertion):
            ac2 += self.insertion[i]
            if unif2 < ac2:
                indexes[1] = i
                break

        # Find improvement
        unif3 = random.random()
        ac3 = 0
        for i in range(self.nImprove):
            ac3 += self.improvement[i]
            if unif3 < ac3:
                indexes[2] = i
                break

        # Remove nodes
        nodes: Set[int] = set()
        if indexes[0] == 0:
            nodes = current.randomRemoval()
        elif indexes[0] == 1:
            nodes = current.shawRemoval()
        elif indexes[0] == 2:
            nodes = current.worstRemoval()
        elif indexes[0] == 3:
            nodes = current.distanceRadialRuin(15.0)
        elif indexes[0] == 4:
            nodes = current.timeRadialRuin(15.0)
        elif indexes[0] == 5:
            nodes = current.distanceRadialRuin(20.0)
        elif indexes[0] == 6:
            nodes = current.timeRadialRuin(20.0)
        elif indexes[0] == 7:
            nodes = current.windowRemoval()
        elif indexes[0] == 8:
            nodes = current.routeRemoval()

        # Insert nodes
        if indexes[1] == 0:
            current.greedyHeuristic(nodes, False)
        elif indexes[1] == 1:
            current.greedyHeuristic(nodes, True)
        elif indexes[1] == 2:
            current.regretHeuristic(2, nodes, False)
        elif indexes[1] == 3:
            current.regretHeuristic(2, nodes, True)
        elif indexes[1] == 4:
            current.regretHeuristic(3, nodes, False)
        elif indexes[1] == 5:
            current.regretHeuristic(3, nodes, True)
        elif indexes[1] == 6:
            current.regretHeuristic(4, nodes, False)
        elif indexes[1] == 7:
            current.regretHeuristic(4, nodes, True)
        elif indexes[1] == 8:
            current.regretHeuristicAll(nodes, False)
        elif indexes[1] == 9:
            current.regretHeuristicAll(nodes, True)

        # Improve solution
        if indexes[2] == 0:
            current.searchShift()
        elif indexes[2] == 1:
            current.searchInterchangeAll()
        elif indexes[2] == 2:
            current.searchOpt2All()
        elif indexes[2] == 3:
            current.crossExchangeAll(7)
        elif indexes[2] == 4:
            current.search2OptInterAll()
        elif indexes[2] == 5:
            current.pathRelocationAll()
        elif indexes[2] == 6:
            current.orOptAll()

        return tuple(indexes)

    def algorithmName(self) -> str:
        return "AdaptiveImproved"

    def getTransitions(self) -> Dict[Tuple[int, int], float]:
        matrix: Dict[Tuple[int, int], float] = {}

        # Removal
        for i in range(self.nRemoval):
            matrix[(0, i)] = round(self.removal[i] * 1E10) / 1E10

        # Insertion
        for i in range(self.nInsertion):
            matrix[(1, i)] = round(self.insertion[i] * 1E10) / 1E10

        # Improvement
        for i in range(self.nImprove):
            matrix[(2, i)] = round(self.improvement[i] * 1E10) / 1E10

        return matrix
