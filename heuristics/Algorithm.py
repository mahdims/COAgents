from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple

class Algorithm(ABC):

    @abstractmethod
    def getParam(self):
        pass

    @abstractmethod
    def setTimeLimit(self, timeLimit: float):
        pass

    @abstractmethod
    def setSeed(self, seed: int):
        pass

    @abstractmethod
    def setIterLimit(self, iterLimit: int):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def getTotalCost(self) -> float:
        pass

    @abstractmethod
    def getOutput(self) -> str:
        pass

    @abstractmethod
    def setDelta(self, delta: float):
        pass

    @abstractmethod
    def getRunTime(self) -> float:
        pass

    @abstractmethod
    def getIterations(self) -> int:
        pass

    @abstractmethod
    def isFeasible(self) -> bool:
        pass

    @abstractmethod
    def setOutput(self, set: bool):
        pass

    @abstractmethod
    def getBestRoutes(self) -> List[List[int]]:
        pass

    @abstractmethod
    def getBestTimes(self) -> List[List[float]]:
        pass

    @abstractmethod
    def algorithmName(self) -> str:
        pass

    @abstractmethod
    def getTransitions(self) -> Dict[Tuple[int, int], float]:
        pass

    @abstractmethod
    def getApply(self) -> Dict[int, float]:
        pass
