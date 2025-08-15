from Algorithm import Algorithm
from VRPSolution import Solution
from ParameterReader import ParameterReader
import random
import math
import time
import copy
import sys

class AdaptiveLargeNeighborhoodSearch(Algorithm):
    def __init__(self, param: ParameterReader):
        self.param = param
        self.set_output = True
        self.timeLimit = 90.0
        self.iterLimit = sys.maxsize
        self.nRemoval = 9
        self.nInsertion = 10
        self.best = Solution(param)
        self.initiated = 0
        self.status = False
        self.bestValue = float('inf')
        self.time = 0.0
        self.output = ""
        self.app = 0
        self.rand = random.Random()
        self.removal = {}
        self.insertion = {}
        self.sigma = [33.0, 9.0, 13.0]
        self.hashValues = []
        self.r = 0.1
        self.history = []
        self.BKS = -1
        self.verbos =  0
        self.tol = 0.001
        

    def setSeed(self, seed: int):
        self.rand.seed(seed)

    def setTimeLimit(self, timeLimit: float):
        self.timeLimit = timeLimit

    def setIterLimit(self, iterLimit: int):
        self.iterLimit = iterLimit
        
    def setBKS(self, BKS: float):
        self.BKS =  BKS
        
    def setInitalSol(self, sol):
        self.best = sol
        self.initiated = 1
        
    def setDelta(self, delta: float):
        pass
    
    
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

    
    def solve(self):
        self.best.setRandom(self.rand)
        self.print(self.param.info() + "\n\n")
        self.print(self.algorithmLine() + "\n")
        self.print(f"running {self.algorithmName()} \n")
        self.print(self.algorithmLine() + "\n\n")

        if self.initiated ==0:
            self.best.savingsMethod()
        self.bestValue = self.best.getTotalCost()
        current = Solution(self.param)
        current.copySolution(self.best)
        currentCost = current.getTotalCost()
        temperature = -self.bestValue * 0.05 / 100 / math.log(0.5)

        self.initWeights()
        totals, scores = self.initMapping()
        self.hashValues.append(current.hashValue())
        iterEndSol = 0
        iterStartSol = 0
        
        self.print(self.toTable("time", 12) + "|" + self.toTable("apply", 10) + "|" +
                   self.toTable("best", 18) + "|" + self.toTable("current", 18) + "|" +
                   self.toTable("tmp", 18) + "|" + self.toTable("sequence", 20))
        self.print("-" * 92 + "\n")

        start_time = time.time()
        self.app = 0
    
        while self.time < self.timeLimit and self.app < self.iterLimit and gap(self.BKS, self.bestValue ) > self.tol:
            self.app += 1
            tmp = Solution(self.param)
            tmp.copySolution(current)
            indexes = self.applyHeuristics(tmp)
            tmpCost = tmp.getTotalCost()
            code = tmp.hashValue()
            type_update = -1

            iterEndSol += 1 
            self.history.append([iterStartSol, iterEndSol, indexes, {iterEndSol: copy.deepcopy(tmp)}])
            
            if code not in self.hashValues:
                self.hashValues.append(code)
            # else:
            #     iterEndSol = self.hashValues.index(code)
            #     self.history.append([iterStartSol, iterEndSol, indexes, {iterEndSol: copy.deepcopy(tmp)}])

                
            if tmpCost < self.bestValue:
                self.best.copySolution(tmp)
                self.bestValue = tmpCost
                current.copySolution(tmp)
                currentCost = current.getTotalCost()
                iterStartSol = iterEndSol
                type_update = 0
            elif tmpCost < currentCost:
                current.copySolution(tmp)
                currentCost = current.getTotalCost()
                iterStartSol = iterEndSol
                type_update = 1
            else:
                prob = math.exp(-1.0 / temperature * (tmpCost - self.bestValue))
                if self.rand.random() < prob and prob!=1:
                    current.copySolution(tmp)
                    currentCost = current.getTotalCost()
                    iterStartSol = iterEndSol
                    type_update = 2

            
            self.updateScores(indexes, type_update, totals, scores)
            self.time = time.time() - start_time
            self.print(self.toTable(f"{self.time:.2f}", 12) + "|" +
                       self.toTable(str(self.app), 10) + "|" +
                       self.toTable(f"{self.bestValue:.2f}", 18) + "|" +
                       self.toTable(f"{currentCost:.2f}", 18) + "|" +
                       self.toTable(f"{tmpCost:.2f}", 18) + "|" +
                       self.toTable(str(indexes), 20) + "\n")
            temperature *= 0.99975

            if (self.app + 1) % 100 == 0:
                self.updateProbabilities(totals, scores)
                totals, scores = self.initMapping()

        self.status = self.best.isFeasible()
        self.print("-" * 92 + "\n\n")
        if self.verbos >= 1:
            self.printSolution()

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

    
    
    def applyHeuristics(self, current : Solution):
        indexes = [None, None]

        # find removal
        unif1 = self.rand.random()
        acc = 0.0
        for i in range(self.nRemoval):
            acc += self.removal[i]
            if unif1 < acc:
                indexes[0] = i
                break
        
        # find insertion
        unif2 = self.rand.random()
        acc = 0.0
        for i in range(self.nInsertion):
            acc += self.insertion[i]
            if unif2 < acc:
                indexes[1] = i
                break
        
        nodes = set()
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

        return indexes

    def getTotalCost(self) -> float:
        if self.status:
            return round(self.best.getTotalCost() * 1E5) / 1E5
        return float('inf')

    def getParam(self) -> ParameterReader:
        return self.param 
     
    def getOutput(self) -> str:
        return self.output

    def getRunTime(self) -> float:
        return round(self.time * 1E2) / 1E2

    def getIterations(self) -> int:
        return self.app

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

    def algorithmLine(self) -> str:
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

    def algorithmName(self) -> str:
        return "AdaptiveLargeNeighborhoodSearch"

    def getTransitions(self) -> dict[tuple[int, int], float]:
        matrix = {}
        # removal
        for i in range(self.nRemoval):
            matrix[(0, i)] = round(self.removal[i] * 1E10) / 1E10
        # insertion
        for i in range(self.nInsertion):
            matrix[(1, i)] = round(self.insertion[i] * 1E10) / 1E10
        return matrix

    def getApply(self) -> dict[int, float]:
        return None



def gap(value1, value2):
    if value1 == -1:
        return 100
    return (value2 - value1) / value1