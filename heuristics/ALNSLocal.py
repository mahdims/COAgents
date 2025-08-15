from collections import defaultdict
from typing import Dict, Set, Tuple
from ALNS import AdaptiveLargeNeighborhoodSearch
from ParameterReader import ParameterReader


class ALNSLocal(AdaptiveLargeNeighborhoodSearch):
    
    def __init__(self, param: ParameterReader):
        super().__init__(param)
        self.improvement: Dict[int, float] = {}
        self.nImprove = 19
    
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

    def applyHeuristics_old(self, current) -> Tuple[int, int, int]:
        # Indexes
        index = -1
        # Find improvement
        unif3 = self.rand.random()
        ac3 = 0
        for i in range(self.nImprove):
            ac3 += self.improvement[i]
            if unif3 < ac3:
                index = i
                break

        # Local Searches
        if index == 0:
            current.searchShift()
        elif index == 1:
            current.searchInterchangeAll()
        elif index == 2:
            current.searchOpt2All()
        elif index == 3:
            current.crossExchangeAll(7)
        elif index == 4:
            current.search2OptInterAll()
        elif index == 5:
            current.pathRelocationAll()
        elif index == 6:
            current.orOptAll()

        return (0,0,index)

    def applyHeuristics(self, current) -> Tuple[int, int, int]:
        # select operator index
        index = -1
        unif3 = self.rand.random()
        ac3 = 0
        for i in range(self.nImprove):
            ac3 += self.improvement[i]
            if unif3 < ac3:
                index = i
                break

        # apply pyvrp local searches
        if index == 0:
            current.searchShift()
        elif index == 1:
            current.searchInterchangeAll()
        elif index == 2:
            current.searchOpt2All()
        elif index == 3:
            current.crossExchangeAll(7)
        elif index == 4:
            current.search2OptInterAll()
        elif index == 5:
            current.pathRelocationAll()
        elif index == 6:
            current.orOptAll()
        elif index == 7:
            current.exchange10()
        elif index == 8:
            current.exchange20()
        elif index == 9:
            current.exchange30()
        elif index == 10:
            current.exchange11()
        elif index == 11:
            current.exchange21()
        elif index == 12:
            current.exchange31()
        elif index == 13:
            current.exchange22()
        elif index == 14:
            current.exchange32()
        elif index == 15:
            current.exchange33()
        elif index == 16:
            current.swap_tails()
        elif index == 17:
            current.swap_routes()
        elif index == 18:
            current.swap_star()
            
        return (0, 0, index)

    def algorithmName(self) -> str:
        return "AdaptiveLocalSearch"

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
