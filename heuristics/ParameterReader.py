from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple
import math
from pyvrp import Model, CostEvaluator
import numpy as np
SCALE_FACTOR = 100000
to_int = lambda x: int(x * SCALE_FACTOR + 0.5)
class ParameterReader(ABC):

    def __init__(self, instance: str):
        self.instance = instance
        self.nVehicles = 0
        self.nNodes = 0
        self.capacity = 0.0
        self.depot = 0
        self.demand: Dict[int, float] = {}
        self.positions: Dict[int, List[float]] = {}
        self.windows: Dict[int, List[float]] = {}
        self.service: Dict[int, float] = {}
        self.time: Dict[Tuple[int, int], float] = {}
        self.typeDouble = False
        self.pyVRPData  = None

    @abstractmethod
    def data(self) -> bool:
        pass

    def read(self) -> bool:
        status = self.data()
        self.computeTime()
        return status

    def setDistanceType(self, typeDouble: bool):
        self.typeDouble = typeDouble

    def computeTime(self):
        # all indexes
        nodes: Set[int] = set(self.positions.keys())

        # create a matrix with all distances
        self.time = {}

        # distance computation, round to the second decimal digit
        for n1 in nodes:
            pos1 = self.positions[n1]
            for n2 in nodes:
                pos2 = self.positions[n2]
                value = self.distancePrecision(math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))
                self.time[(n1, n2)] = value

    def distancePrecision(self, distance: float) -> float:
        if self.typeDouble:
            return distance
        else:
            return math.floor(distance * 1E1) / 1E1

    def getTimeWindows(self) -> Dict[int, List[float]]:
        return dict(self.windows)

    def getTime(self) -> Dict[Tuple[int, int], float]:
        return dict(self.time)

    def getService(self) -> Dict[int, float]:
        return dict(self.service)

    def getDemand(self) -> Dict[int, float]:
        return dict(self.demand)

    def getNVehicles(self) -> int:
        return self.nVehicles

    def getCapacity(self) -> float:
        return self.capacity

    def getNodes(self) -> Set[int]:
        return set(self.positions.keys())

    def getInstance(self) -> str:
        return self.instance

    def getDepot(self) -> int:
        return self.depot

    def clear(self):
        # clear maps
        self.demand.clear()
        self.positions.clear()
        self.windows.clear()
        self.service.clear()

        # clear matrix
        self.time.clear()

        self.instance = None

    @abstractmethod
    def info(self) -> str:
        pass
    
    
    def BuildPyVRPData(self):
        """
        Build a PyVRP ProblemData instance from a ParameterReader.
        """
        m = Model()

        # 1. Vehicle type
        m.add_vehicle_type(self.nNodes, capacity=[int(self.getCapacity())])

        # 2. Add depots and clients, track mapping from param IDs to Model.Location
        depot_id = self.getDepot()
        positions = self.positions
        windows = self.getTimeWindows()
        service = self.getService()
        demand = self.getDemand()

        id_to_loc = {}

        # add depot
        x_dep, y_dep = positions[depot_id]
        tw_early, tw_late = windows[depot_id]
        m.add_depot(
            x=to_int(x_dep), y=to_int(y_dep),
            tw_early=to_int(tw_early), tw_late=to_int(tw_late),
            name=str(depot_id)
        )
        id_to_loc[depot_id] = m.locations[-1]

        # add clients
        for node in sorted(self.getNodes()):
            if node == depot_id:
                continue
            x, y = positions[node]
            tw_early, tw_late = windows[node]
            m.add_client(
                x=to_int(x), y=to_int(y),
                delivery=int(demand[node]),
                service_duration=to_int(service[node]),
                tw_early=to_int(tw_early), tw_late=to_int(tw_late),
                name=str(node)
            )
            id_to_loc[node] = m.locations[-1]

        # 3. Register travel times via edges
        node_ids = [depot_id] + [n for n in sorted(self.getNodes()) if n != depot_id]
        for u in node_ids:
            for v in node_ids:
                t = self.time[(u, v)]
                m.add_edge(
                    id_to_loc[u],
                    id_to_loc[v],
                    to_int(t),
                    duration=to_int(t)
                )

        # 4. Build and return ProblemData
        self.pyVRPData = m.data()
        return self.pyVRPData
