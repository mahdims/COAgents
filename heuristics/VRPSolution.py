import collections
from ParameterReader import ParameterReader
import Utils as ut
import json
import copy
import math
import time
from pyvrp import Solution as VRPSolution, ProblemData, RandomNumberGenerator, CostEvaluator
from pyvrp.search import (
    LocalSearch, compute_neighbours,
    Exchange10, Exchange20, Exchange30, Exchange11, Exchange21, Exchange31,
    Exchange22, Exchange32, Exchange33, SwapTails,
    SwapRoutes, SwapStar
)

# The solution decoder
def stringToRoutes(routesStr : str) -> tuple[ float, list ] :
    """
    Converts a string representation of routes back to a list of routes and extracts the cost if available.
    Args:
        routesStr (str): A string representation of the routes.
    """
    routes = []
    cost = None
    for line in routesStr.strip().split('\n'):
        if line.startswith("Route #"):
            route = list(map(int, line.split(':')[1].strip().split()))
            routes.append(route)
        elif line.startswith("Cost"):
            cost = float(line.split()[1])
    return (cost, routes)

class Solution:
    def __init__(self, param: ParameterReader):
        self.Id = '0'
        # Parameters
        self.param = param
        self.depot = param.getDepot()
        
        self.nlowlevel = 27
        self.maxSetSize = 40
        self.rand = None
        # Get parameters
        self.times = ut.EfficientTime(param.getTime())
        self.windows = param.getTimeWindows()
        self.service = param.getService()
        self.demand = param.getDemand()
        
        # Solution routes 
        self.routes = []
        for _ in range(param.getNVehicles()):
            # Dummy route
            route = [self.depot, self.depot]
            self.routes.append(route)
        
        # Compute penalty constant
        total = 1
        nodes = param.getNodes() - {self.depot}
        for node in nodes:
            total += 2 * self.times[self.depot, node]
        self.constFeas = math.ceil(total * (len(nodes) + 1))
        
        # Starting values
        self.totalCost = self.constFeas # large penalty
        self.feasible = False
        
        # Lists
        self.feasibleRoutes = [False] * len(self.routes)
        self.routesTimes = [None] * len(self.routes)
        self.routesLoads = [0.0] * len(self.routes)
        self.routesCosts = [self.constFeas] * len(self.routes)
        
        # Adding penalties to some edges
        self.penalizedEdges = []
    
    def addPenalizedEdges(self, edges):
        if isinstance(edges, tuple):
            self.penalizedEdges.append(edges)
        elif isinstance(edges, list):
            self.penalizedEdges.extend(edges)
        else:
            print("ERROR in edge type")
        
    def calEdgePenalty(self, addedEdges=[], removedEdges=[]):
        
        if self.penalizedEdges == []: 
            return 0
        penatly  = 0
        if addedEdges: 
            for edge in addedEdges: 
                if edge in self.penalizedEdges: 
                    penatly += self.constFeas
        
        if removedEdges: 
            for edge in removedEdges:
                if edge in self.penalizedEdges: 
                    penatly -= self.constFeas
        
        return penatly
    
    def copySolution(self, tmp):
        
        self.Id = tmp.Id
        # Copy routes
        self.routes = [copy.deepcopy(route) for route in tmp.routes]
        
        # Copy solution
        self.totalCost = tmp.totalCost
        self.feasible = tmp.feasible
        
        # Copy lists
        self.feasibleRoutes = copy.deepcopy(tmp.feasibleRoutes)
        self.routesTimes = [copy.deepcopy(times) if times else None for times in tmp.routesTimes]
        self.routesLoads = copy.deepcopy(tmp.routesLoads)
        self.routesCosts = copy.deepcopy(tmp.routesCosts)
        
        self.rand  = tmp.rand
        self.penalizedEdges = copy.deepcopy(tmp.penalizedEdges)

    def readRoutesFromSol(self, file_path):
        """
        Reads routes from a .sol file and converts them to a list of lists.
        Args:
            file_path (str): Path to the .sol file.
        Returns:
            list: A list of routes, where each route is represented as a list of nodes.
        """
        self.routes = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Route #"):
                    route = [self.depot] + list(map(int, line.split(':')[1].strip().split())) + [self.depot]
                    self.routes.append(route)
                elif line.startswith("Cost"):
                    self.totalCost = float(line.split()[1])    
        self.updateSolution()
    
    def writeToSol(self, path):
        """
        Writes routes to a .sol file.
        Args:
            file_path (str): Path to the .sol file.
            routes (list): A list of routes, where each route is represented as a list of nodes.
            cost (float): The cost of the solution.
        """
        with open(path + f"{self.param.instance}.sol", 'w') as file:
            for idx, route in enumerate(self.routes, start=1):
                route_str = " ".join(map(str, route))
                file.write(f"Route #{idx}: {route_str}\n")
            file.write(f"Cost {self.totalCost}\n")
    
    
    def writeToJson(self, path):
        """
        Saves the Routes to a JSON file.
        """
        outDic =  {'routes': self.routes}
        with open(path + f"{self.param.instance}_bSol", 'w') as file:
            json.dump(outDic, file, indent=4)
    
    def routesToString(self):
        """
        Converts the list of routes to a string format, including the cost if available.
        Returns:
            str: A string representation of the routes.
        """
        routesStr = ""
        for i, route in enumerate(self.routes, 1):
            routesStr += f"Route #{i}: {' '.join(map(str, route))}\n"
        if self.totalCost is not None:
            routesStr += f"Cost {self.totalCost}\n"
        return routesStr

    def stringToRoutes(self, routesStr : str) -> None :
        """
        Converts a string representation of routes back to a list of routes and extracts the cost if available.
        Args:
            routesStr (str): A string representation of the routes.
        """
        (self.totalCost, self.routes) = stringToRoutes(routesStr)
        self.updateSolution()

    def setWithRoutes(self, routes: list[list]):
        self.routes  = routes  
        self.updateSolution()
        
    def setTimes(self, times):
        self.times = ut.EfficientTime(times)
        self.updateSolution()
    
    def updateSolution(self):
        # Remove empty routes
        self.routes = [route for route in self.routes if len(route) >= 3]
        
        # Add additional empty route if there are vehicles available
        if self.param.getNVehicles() - len(self.routes) > 0:
            self.routes.append([self.depot, self.depot])
        
        # Starting values
        self.totalCost = 0.0
        self.feasible = True
        
        # Lists
        self.feasibleRoutes = [True] * len(self.routes)
        self.routesTimes = [None] * len(self.routes)
        self.routesLoads = [0.0] * len(self.routes)
        self.routesCosts = [0.0] * len(self.routes)
               
        # Update routes
        for i in range(len(self.routes)):
            self.updateRoute(i)
        
        # Check number of vehicles
        if self.param.getNVehicles() < len(self.routes):
            self.feasible = False
            self.totalCost += self.constFeas * (len(self.routes) - self.param.getNVehicles())
        
        # Check all routes
        for cost, feasible in zip(self.routesCosts, self.feasibleRoutes):
            self.totalCost += cost
            if not feasible:
                self.feasible = False
                self.totalCost += self.constFeas
        
        # Nodes must be visited exactly once
        nodes = self.param.getNodes() - {self.depot}
        for route in self.routes:
            for node in route[1:-1]:
                if node not in nodes:
                    self.feasible = False
                    self.totalCost += self.constFeas
                nodes.discard(node)
        
        if nodes:
            self.feasible = False
            self.totalCost += self.constFeas * len(nodes)

    def updateRoute(self, index):
        route = self.routes[index]
        cost = 0.0
        last = route[0]
        time = self.service[last]
        quantity = 0.0
        locTimes = [[time, time, time]]
        
        for i in range(1, len(route)):
            vector = [0.0] * 3
            node = route[i]
            cost += self.times[last, node]
            time += self.times[last, node]
            
            vector[0] = time
            if self.windows[node][1] < time:
                cost += self.constFeas
            
            if self.windows[node][0] > time:
                time = self.windows[node][0]
            vector[1] = time
            
            quantity += self.demand[node]
            if quantity > self.param.getCapacity():
                cost += self.constFeas
            
            time += self.service[node]
            vector[2] = time
            locTimes.append(vector)
            last = node
        
        if len(self.routesCosts) > index:
            self.routesCosts[index] = cost
            self.routesLoads[index] = quantity
            self.feasibleRoutes[index] = cost < self.constFeas
            self.routesTimes[index] = locTimes
        else:
            self.routesCosts.append(cost)
            self.routesLoads.append(quantity)
            self.feasibleRoutes.append(cost < self.constFeas)
            self.routesTimes.append(locTimes)

    def setRandom(self, rand):
        self.rand =  rand
    
    def isFeasible(self):
        return self.feasible

    def getTotalCost(self):
        return self.totalCost

    def getRoutes(self):
        return [route for route in self.routes if len(route) > 2]

    def hashValue(self):
        return hash(tuple(tuple(route) for route in sorted(self.routes)))


    def getEdges(self):
        edges  = []
        for route in self.routes :
            if len(route) <= 2:
                continue
            
            cNode= route[0]
            for node in route[1:]:
                edges.append( (cNode, node) )
                cNode =node
        return edges
    
    def checkFeasible(self, route):
        last = route[0]
        time = self.service[last]
        quantity = 0.0
        
        for i in range(1, len(route)):
            node = route[i]
            
            # Check time window
            time += self.times[last, node]
            if self.windows[node][1] < time:
                return False
            
            # Wait if too early
            if self.windows[node][0] > time:
                time = self.windows[node][0]
            
            # Check capacity
            quantity += self.demand[node]
            if quantity > self.param.getCapacity():
                return False
            
            # Add service time
            time += self.service[node]
            last = node
        
        return True

    def getRouteTimes(self):
        arrival = []
        for times in self.routesTimes:
            if times and len(times) >= 3:
                tmp = [vector[0] for vector in times]
                arrival.append(tmp)
        return arrival

    def outAndBackRoutes(self):
        # Initialize routes as out-and-back
        self.routes.clear()
        nodes = self.param.getNodes() - {self.depot}
        for node in nodes:
            route = [self.depot, node, self.depot]
            self.routes.append(route)
        
        self.updateSolution()

    def savingsMethod(self):
        # Initialize routes as out-and-back, then merge
        self.outAndBackRoutes()
        self.mergeRoutes()
        self.updateSolution()

    def mergeRoutes(self):
        # Create savings
        self.computeSavings()
        improv = True
        
        while improv:
            improv = False
            check = {route[1]: i for i, route in enumerate(self.routes) if len(route) >= 3}
            indexes = list(range(len(self.routes)))
            self.rand.shuffle(indexes)
            
            for index1 in indexes:
                route = self.routes[index1]
                if len(route) < 3:
                    continue
                
                node = route[-2]
                options = list(self.savings.get(node, {}).keys())
                for con in options:
                    self.savings[node].pop(con, None)
                    if con not in check or check[con] == index1:
                        continue
                    
                    index2 = check[con]
                    load1 = self.routesLoads[index1]
                    load2 = self.routesLoads[index2]
                    if load1 + load2 > self.param.getCapacity():
                        continue
                    
                    route1 = self.routes[index1]
                    route2 = self.routes[index2]
                    time = self.routesTimes[index1][-1][2]
                    last = node
                    timeFeasible = True
                    
                    for j in range(1, len(route2)):
                        next_node = route2[j]
                        time += self.times[last, next_node]
                        if self.windows[next_node][1] < time:
                            timeFeasible = False
                            break
                        
                        if self.windows[next_node][0] > time:
                            time = self.windows[next_node][0]
                        time += self.service[next_node]
                        last = next_node
                    
                    if not timeFeasible:
                        continue
                    
                    new_route = route1[:-1] + route2[1:]
                    self.routes[index1] = new_route
                    self.routes[index2] = []
                    self.updateRoute(index1)
                    
                    check.pop(con, None)
                    if con in self.savings:
                        self.savings[con].pop(node, None)
                    improv = True
                    break
            
            self.updateSolution()

    def computeSavings(self):
        self.savings = {}
        for i, route1 in enumerate(self.routes):
            if len(route1) < 3:
                continue
            
            node1 = route1[-2]
            savings_map = {}
            for j, route2 in enumerate(self.routes):
                if i == j or len(route2) < 3:
                    continue
                
                node2 = route2[1]
                value = self.times[node1, self.depot] + self.times[self.depot, node2] - self.times[node1, node2]
                savings_map[j] = value
            
            sorted_savings = dict(sorted(savings_map.items(), key=lambda item: item[1], reverse=True))
            self.savings[node1] = sorted_savings

    def searchShift(self, all = False):
        if all: 
            indexes = list(range(len(self.routes)))
        else:
            indexes = self.getRandomIndexes()
        for index in indexes:
            self.searchShiftOther(index)
            self.searchShiftSame(index)
        self.updateSolution()

    def getRandomIndexes(self):
        indexes = list(range(len(self.routes)))
        self.rand.shuffle(indexes)
        total = min(len(indexes), self.maxSetSize)
        return indexes[:total]

    def searchShiftOther(self, index):
        # get route
        route = self.routes[index]

        # check number of nodes
        if len(route) < 3:
            return

        # info
        capacity = self.param.getCapacity()

        # all nodes
        k = 1
        while k < len(route) - 1:
            # get nodes
            node = route[k]
            nodeLoad = self.demand[node]
            latest = self.windows[node][1]
            n1 = route[k - 1]
            n2 = route[k + 1]

            # check routes in random order
            indexes = self.getRandomIndexes()

            # get list all routes
            inserted = False
            for i in range(len(indexes)):
                
                if (inserted):
                    break
                
                index2 = indexes[i]
                load = self.routesLoads[index2]
                if index2 == index or load + nodeLoad > capacity:
                    continue

                # check all positions
                review = self.routes[index2][:]
                for j in range(1, len(review)):
                    m1 = review[j - 1]
                    m2 = review[j]

                    # compute diff
                    org = self.times.get(n1, node) + self.times.get(node, n2) + self.times.get(m1, m2)
                    new = self.times.get(n1, n2) + self.times.get(m1, node) + self.times.get(node, m2)
                    penalty = self.calEdgePenalty(removedEdges=[(n1, node), (node, n2), (m1, m2)], 
                                                  addedEdges=[(n1, n2), (m1, node), (node, m2)])
                    
                    if org - new - penalty <= 0:
                        continue

                    # check time
                    lastTime = self.routesTimes[index2][j - 1][2]
                    if lastTime + self.times.get(m1, node) > latest:
                        break

                    # new route
                    tmp2 = review[:]
                    tmp2.insert(j, node)
                    if self.checkFeasible(tmp2):
                        # update route
                        self.routes[index2] = tmp2
                        self.updateRoute(index2)

                        # remove node
                        tmp1 = route[:]
                        tmp1.remove(node)
                        self.routes[index] = tmp1
                        self.updateRoute(index)

                        # update
                        route = self.routes[index][:]
                        inserted = True
                        break

            # in this case increase k
            if not inserted:
                k += 1

    def searchShiftSame(self, index):
        # get route
        route = self.routes[index]

        # check number of nodes
        if len(route) < 4:
            return

        # all nodes
        k = 1
        while k < len(route) - 1:
            # get nodes
            node = route[k]
            latest = self.windows[node][1]
            n1 = route[k - 1]
            n2 = route[k + 1]

            pos = -1

            # check all positions
            review = route[:]
            review.remove(node)
            for j in range(1, len(review)):
                m1 = review[j - 1]
                m2 = review[j]

                # compute diff
                org = self.times.get(n1, node) + self.times.get(node, n2) + self.times.get(m1, m2)
                new = self.times.get(n1, n2) + self.times.get(m1, node) + self.times.get(node, m2)
                
                penalty = self.calEdgePenalty(removedEdges=[(n1, node), (node, n2), (m1, m2)], 
                                                  addedEdges=[(n1, n2), (m1, node), (node, m2)])
                    
                if org - new - penalty <= 0:
                    continue

                # check time
                lastTime = self.routesTimes[index][j - 1][2]
                if lastTime + self.times.get(m1, node) > latest:
                    break

                # new route
                tmp = review[:]
                tmp.insert(j, node)
                if self.checkFeasible(tmp):
                    # update route
                    self.routes[index] = tmp
                    self.updateRoute(index)

                    # update
                    route = self.routes[index][:]
                    pos = j
                    break

            # in this case increase k
            if pos <= k:
                k += 1
    
    def crossExchange(self, index, length):
        # get route
        route1 = self.routes[index]
        feas1 = self.checkFeasible(route1)

        # check number of nodes
        if len(route1) < 3:
            return

        # check routes randomly
        indexes = self.getRandomIndexes()
        for k in indexes:
            if k == index:
                continue
            route2 = self.routes[k][:]
            feas2 = self.checkFeasible(route2)

            # check combinations
            for i1 in range(len(route1) - 2):
                for i2 in range(len(route2) - 2):
                    for j1 in range(i1, min(len(route1) - 1, i1 + length)):
                        for j2 in range(i2, min(len(route2) - 1, i2 + length)):
                            # get nodes route1
                            w1 = route1[i1]
                            x1 = route1[i1 + 1]
                            y1 = route1[j1]
                            z1 = route1[j1 + 1]

                            # get nodes route2
                            w2 = route2[i2]
                            x2 = route2[i2 + 1]
                            y2 = route2[j2]
                            z2 = route2[j2 + 1]

                            # compute gain
                            org = self.times.get(w1, x1) + self.times.get(w2, x2) + self.times.get(y1, z1) + self.times.get(y2, z2)
                            change = self.times.get(w1, x2) + self.times.get(w2, x1) + self.times.get(y1, z2) + self.times.get(y2, z1)
                            
                            penalty = self.calEdgePenalty(removedEdges=[(w1, x1), (w2, x2), (y1, z1), (y2, z2)], 
                                                  addedEdges=[(w1, x2), (w2, x1), (y1, z2), (y2, z1)])
                            diff = org - change - penalty
                            if diff > 0 or not feas1 or not feas2:
                                # create routes
                                tmp1 = route1[:i1 + 1] + route2[i2 + 1:j2 + 1] + route1[j1 + 1:]
                                tmp2 = route2[:i2 + 1] + route1[i1 + 1:j1 + 1] + route2[j2 + 1:]

                                if (not feas1 or self.checkFeasible(tmp1)) and (not feas2 or self.checkFeasible(tmp2)):
                                    self.routes[index] = tmp1
                                    self.routes[k] = tmp2
                                    return

    def crossExchangeAll(self, length, all= False):
        
        if all: 
            indexes = list(range(len(self.routes)))
        else:
            # check routes randomly
            indexes = self.getRandomIndexes()
        for i in range(len(indexes)):
            self.crossExchange(indexes[i], length)

        self.updateSolution()

    def searchOpt2All(self):
        for i in range(len(self.routes)):
            self.searchOpt2(i)
        
        self.updateSolution()
    
    
    def searchOpt2(self, index):
        # get route
        route = self.routes[index]
        length = len(route)
        feas = self.feasibleRoutes[index]

        # check number of nodes
        if length < 3:
            return

        # iterate while there is improvement
        improv = True
        while improv:
            improv = False

            # check all combinations 0 < i < j < len - 1
            for i in range(1, length - 2):
                insertNotFeasible = False
                for j in range(i + 1, length - 1):
                    if insertNotFeasible:
                        break

                    # compute times
                    arrival = self.routesTimes[index][i][2]

                    # new route
                    tmp = route[:i]
                    for k in range(j, i - 1, -1):
                        tmp.append(route[k])

                        # check time window
                        arrival += self.times.get(route[k - 1], route[k])
                        if self.windows[route[k]][0] > arrival:
                            arrival = self.windows[route[k]][0]

                        # break
                        if self.windows[route[k]][1] < arrival:
                            insertNotFeasible = True
                            break

                    tmp.extend(route[j + 1:])

                    # if better and feasible, keep it
                    org = self.times.get(route[i - 1], route[i]) + self.times.get(route[j], route[j + 1])
                    change = self.times.get(route[i - 1], route[j]) + self.times.get(route[i], route[j + 1])
                    
                    penalty = self.calEdgePenalty(removedEdges=[(route[i - 1], route[i]), (route[j], route[j + 1])], 
                                                  addedEdges=[(route[i - 1], route[j]), (route[i], route[j + 1])])
                    diff = org - change - penalty
                    if diff > 0 and not insertNotFeasible:
                        # new solution with changes
                        if not feas or self.checkFeasible(tmp):
                            # update
                            self.routes[index] = tmp
                            self.updateRoute(index)
                            route = tmp[:]
                            feas = self.feasibleRoutes[index]
                            # keep iterating
                            improv = True

    def orOptAll(self):
        # check all vehicles
        for i in range(len(self.routes)):
            self.orOpt(i)

        self.updateSolution()

    def orOpt(self, index):
        # get route
        route = self.routes[index]

        # check number of nodes
        if len(route) < 6:
            return

        # all nodes
        for i in range(1, len(route) - 4):
            n1 = route[i]
            n2 = route[i + 1]
            for j in range(i + 1, len(route) - 3):
                n3 = route[j]
                n4 = route[j + 1]
                for k in range(j + 1, len(route) - 2):
                    n5 = route[k]
                    n6 = route[k + 1]

                    # compute diff
                    org = self.times.get(n1, n2) + self.times.get(n3, n4) + self.times.get(n5, n6)
                    change = self.times.get(n1, n4) + self.times.get(n5, n2) + self.times.get(n3, n6)
                    penalty = self.calEdgePenalty(removedEdges=[(n1, n2), (n3, n4), (n5, n6)], 
                                                  addedEdges=[(n1, n4), (n5, n2), (n3, n6)])
                    if org - change - penalty<= 0:
                        continue

                    # new route
                    tmp = route[:i + 1] + route[j + 1:k + 1] + route[i + 1:j + 1] + route[k + 1:]
                    if self.checkFeasible(tmp):
                        # update route
                        self.routes[index] = tmp
                        self.updateRoute(index)
                        return   


    def searchInterchange(self, index1, index2):
        # less than 1 route, return
        if len(self.routes) < 2:
            return

        # info
        capacity = self.param.getCapacity()

        # get the two routes
        route1 = self.routes[index1]
        route2 = self.routes[index2]
        len1 = len(route1)
        len2 = len(route2)
        feas1 = self.feasibleRoutes[index1]
        feas2 = self.feasibleRoutes[index2]

        if len1 < 3 or len2 < 3:
            return

        restart = True  # Variable to track if we need to restart
        while restart:
            restart = False  # Reset restart flag at the beginning of each iteration
            # all combinations of i and j
            for i in range(1, len1 - 1):
                for j in range(1, len2 - 1):
                    # check load
                    load1 = self.routesLoads[index1] - self.demand[route1[i]] + self.demand[route2[j]]
                    load2 = self.routesLoads[index2] - self.demand[route2[j]] + self.demand[route1[i]]
                    if load1 > capacity or load2 > capacity:
                        continue

                    # check times
                    time1 = self.routesTimes[index1][i - 1][2] + self.times.get(route1[i - 1], route2[j])
                    time2 = self.routesTimes[index2][j - 1][2] + self.times.get(route2[j - 1], route1[i])
                    if time1 > self.windows[route2[j]][1] or time2 > self.windows[route1[i]][1]:
                        continue

                    # exchange nodes
                    tmp1 = route1[:]
                    tmp1[i] = route2[j]
                    tmp2 = route2[:]
                    tmp2[j] = route1[i]

                    # if total cost is less than best and both feasible, then keep them
                    org = (self.times.get(route1[i - 1], route1[i]) + self.times.get(route1[i], route1[i + 1]) +
                        self.times.get(route2[j - 1], route2[j]) + self.times.get(route2[j], route2[j + 1]))
                    change = (self.times.get(route1[i - 1], route2[j]) + self.times.get(route2[j], route1[i + 1]) +
                            self.times.get(route2[j - 1], route1[i]) + self.times.get(route1[i], route2[j + 1]))
                    
                    penalty = self.calEdgePenalty(removedEdges=[(route1[i - 1], route1[i]),(route1[i], route1[i + 1]) ,
                                                                (route2[j - 1], route2[j]) , (route2[j], route2[j + 1])], 
                                                  addedEdges=[(route1[i - 1], route2[j]), (route2[j], route1[i + 1]),
                                                              (route2[j - 1], route1[i]) , (route1[i], route2[j + 1])])
                    diff = org - change - penalty
                    if diff > 0:
                        # new solution
                        if (not feas1 or self.checkFeasible(tmp1)) and (not feas2 or self.checkFeasible(tmp2)):
                            # update
                            self.routes[index1] = tmp1
                            self.routes[index2] = tmp2
                            self.updateRoute(index1)
                            self.updateRoute(index2)
                            route1 = tmp1[:]
                            route2 = tmp2[:]
                            feas1 = self.feasibleRoutes[index1]
                            feas2 = self.feasibleRoutes[index2]
                            restart = True
                            break

                if restart:
                    break

    def searchInterchangeAll(self, all=False):
        if all: 
            indexes = list(range(len(self.routes)))
        else:
            indexes = self.getRandomIndexes()
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                self.searchInterchange(indexes[i], indexes[j])

        self.updateSolution()
        
    def pathRelocationAll(self, all = False):
        if all: 
            indexes = list(range(len(self.routes)))
        else:
            indexes = self.getRandomIndexes()
            
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                self.pathRelocation(indexes[i], indexes[j])

        self.updateSolution()

    def pathRelocation(self, index1, index2):
        # info
        capacity = self.param.getCapacity()

        # get the two routes
        route1 = self.routes[index1]
        route2 = self.routes[index2]
        load1 = self.routesLoads[index1]
        load2 = self.routesLoads[index2]

        if len(route1) < 3 or len(route2) < 3:
            return

        # all combinations of i and j
        for i in range(1, len(route1) - 2):
            for j in range(i + 1, len(route1) - 1):
                # check load
                load = 0
                for l in range(i + 1, j + 1):
                    load += self.demand[route1[l]]
                if load + load2 > capacity:
                    break

                # nodes route 1
                n1 = route1[i]
                n2 = route1[i + 1]
                n3 = route1[j]
                n4 = route1[j + 1]

                # check all positions in route2
                for k in range(1, len(route2) - 1):
                    # nodes route 2
                    n5 = route2[k]
                    n6 = route2[k + 1]

                    # compute diff
                    org = self.times.get(n1, n2) + self.times.get(n3, n4) + self.times.get(n5, n6)
                    change = self.times.get(n1, n4) + self.times.get(n5, n2) + self.times.get(n3, n6)
                    
                    penalty = self.calEdgePenalty(removedEdges=[(n1, n2), (n3, n4) , (n5, n6)],
                                                  addedEdges=[(n1, n4), (n5, n2), (n3, n6)])
                        
                    if org - change - penalty <= 0:
                        continue

                    # new routes
                    tmp1 = route1[:i + 1] + route1[j + 1:]
                    tmp2 = route2[:k + 1] + route1[i + 1:j + 1] + route2[k + 1:]
                    if self.checkFeasible(tmp2):
                        # update
                        self.routes[index1] = tmp1
                        self.routes[index2] = tmp2
                        self.updateRoute(index1)
                        self.updateRoute(index2)
                        return
    
    
    
    def search2OptInterAll(self, all =False):
        
        if all:
            indexes = list(range(len(self.routes)))
        else:
            indexes = self.getRandomIndexes()
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                self.search2OptInter(indexes[i], indexes[j])

        self.updateSolution()
      
    def search2OptInter(self, index1, index2):
        # less than 1 route, return
        if len(self.routes) < 2:
            return

        # info
        capacity = self.param.getCapacity()

        # get the two routes
        route1 = self.routes[index1]
        route2 = self.routes[index2]
        feas1 = self.feasibleRoutes[index1]
        feas2 = self.feasibleRoutes[index2]

        if len(route1) < 3 or len(route2) < 3:
            return

        restart = True  # Variable to track if we need to restart
        while restart:
            restart = False 
            # all combinations of i and j
            total1 = 0
            for i in range(len(route1) - 1):
                # keep track of volume
                total1 += self.demand[route1[i]]
                total2 = 0
                for j in range(len(route2) - 1):
                    # check load
                    total2 += self.demand[route2[j]]
                    load1 = self.routesLoads[index2] - total2 + total1
                    load2 = self.routesLoads[index1] - total1 + total2
                    if load1 > capacity or load2 > capacity:
                        continue

                    # check times
                    time1 = self.routesTimes[index1][i][2] + self.times.get(route1[i], route2[j + 1])
                    time2 = self.routesTimes[index2][j][2] + self.times.get(route2[j], route1[i + 1])
                    if time1 > self.windows[route2[j + 1]][1] or time2 > self.windows[route1[i + 1]][1]:
                        continue

                    # exchange nodes
                    tmp1 = route1[:i + 1] + route2[j + 1:]
                    tmp2 = route2[:j + 1] + route1[i + 1:]

                    # if total cost is less than best and both feasible, then keep them
                    org = (self.times.get(route1[i], route1[i + 1]) + self.times.get(route2[j], route2[j + 1]))
                    change = (self.times.get(route1[i], route2[j + 1]) + self.times.get(route2[j], route1[i + 1]))
                    
                    penalty = self.calEdgePenalty(removedEdges=[(route1[i], route1[i + 1]), (route2[j], route2[j + 1])],
                                                  addedEdges=[(route1[i], route2[j + 1]), (route2[j], route1[i + 1])])
                    
                    
                    diff = org - change - penalty
                    if diff > 0:
                        # new solution
                        if (not feas1 or self.checkFeasible(tmp1)) and (not feas2 or self.checkFeasible(tmp2)):
                            # update
                            self.routes[index1] = tmp1
                            self.routes[index2] = tmp2
                            self.updateRoute(index1)
                            self.updateRoute(index2)
                            route1 = tmp1[:]
                            route2 = tmp2[:]
                            feas1 = self.feasibleRoutes[index1]
                            feas2 = self.feasibleRoutes[index2]
                            restart = True
                            break
                if restart:
                    break
                        


    def relatednessMeasure(self, node):
        # measure parameters
        phi = 9.0
        xi = 3.0
        psi = 2.0

        # map with values
        values = {}
        arrival = {}

        # find route
        route = []
        for r in self.routes:
            if node in r:
                route = r
                break

        if not route:
            return None

        # compute time
        last = route[0]
        time = self.service[last]
        for nd in route:
            # consider the arrival time to check time window
            time += self.times.get(last, nd)

            # wait if too early
            if self.windows[nd][0] > time:
                time = self.windows[nd][0]

            # compute time up to node
            if node == nd:
                break

            # finally sum service time
            time += self.service[nd]
            last = nd

        # compute time distance for all nodes
        for route in self.routes:
            last = route[0]
            arr = self.service[last]
            for j in range(1, len(route) - 1):
                # consider the arrival time to check time window
                nd = route[j]
                arr += self.times.get(last, nd)

                # wait if too early
                if self.windows[nd][0] > arr:
                    arr = self.windows[nd][0]

                # include value for node
                arrival[nd] = abs(arr - time)

                # finally sum service time
                arr += self.service[nd]
                last = nd

        # normalize values
        max_distance = self.times.max(node)
        max_arrival = max(arrival.values())

        # compute values
        for nd in arrival:
            value = (phi * self.times.get(node, nd) / max_distance +
                    xi * arrival[nd] / max_arrival +
                    psi * abs(self.demand[nd] - self.demand[node]))
            values[nd] = value

        # sort values
        relate = dict(sorted(values.items(), key=lambda item: item[1]))

        return relate
    
    
    
    
    def shawRemoval(self):
        # nodes
        nodes = list(self.param.getNodes())
        nodes.remove(self.depot)

        # parameters
        p = 6
        limit = round(len(nodes) * 0.4)
        q = 4 + self.rand.randint(0, min(97, limit))

        # random request
        index = self.rand.randint(0, len(nodes) - 1)
        node = nodes[index]

        # list of nodes to remove
        remove = [node]
        values = {}
        while len(remove) < q:
            index = self.rand.randint(0, len(remove) - 1)
            node = remove[index]

            # create list if it is not in the map
            if node in values:
                map_ = values[node]
            else:
                map_ = self.relatednessMeasure(node)

            # remove values in the set
            for tmp in remove:
                map_.pop(tmp, None)
            values[node] = map_

            # sample position
            y = self.rand.random()
            pos = round(len(map_) * (y ** p))
            pos = max(pos - 1, 0)
            list_ = list(map_.keys())
            remove.append(list_[pos])

        # remove all nodes in the list
        for nd in remove:
            # find route
            route = []
            ind = -1
            for i in range(len(self.routes)):
                tmp = list(self.routes[i])
                if nd in tmp:
                    tmp.remove(nd)
                    route = list(tmp)
                    ind = i
                    break

            # set route
            self.routes[ind] = route
            self.updateRoute(ind)

        return set(remove)

    def randomRemoval(self):
        # nodes
        nodes = list(self.param.getNodes())
        nodes.remove(self.depot)

        # parameters
        total = len(nodes)
        limit = round(total * 0.4)
        q = 4 + self.rand.randint(0, min(97, limit))

        # list of nodes to remove
        remove = set()
        while len(remove) < q:
            # random request
            index = self.rand.randint(0, total - 1)
            remove.add(nodes[index])

        # remove all nodes in the list
        for nd in remove:
            # find route
            route = []
            ind = -1
            for i in range(len(self.routes)):
                tmp = list(self.routes[i])
                if nd in tmp:
                    tmp.remove(nd)
                    route = list(tmp)
                    ind = i
                    break

            # set route
            self.routes[ind] = route
            self.updateRoute(ind)

        return remove

    def worstRemoval(self):
        # nodes
        nodes = list(self.param.getNodes())
        nodes.remove(self.depot)

        # parameters
        p = 6
        limit = round(len(nodes) * 0.4)
        q = 4 + self.rand.randint(0, min(97, limit))

        # compute values
        relation = {}
        values = {}
        for i in range(len(self.routes)):
            route = self.routes[i]
            for j in range(1, len(route) - 1):
                node = route[j]
                n1 = route[j - 1]
                n2 = route[j + 1]
                values[node] = self.times.get(n1, node) + self.times.get(node, n2) - self.times.get(n1, n2)
                relation[node] = i

        # sort map
        values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))

        # list of nodes to remove
        remove = []
        while len(remove) < q:
            # sample position
            y = self.rand.random()
            pos = round(len(values) * (y ** p))
            pos = max(pos - 1, 0)
            tmp = list(values.keys())
            node = tmp[pos]
            remove.append(node)

            # remove node from route
            index = relation[node]
            route = self.routes[index]
            route.remove(node)
            self.routes[index] = route
            self.updateRoute(index)

            # update map
            values.pop(node, None)
            for i in range(1, len(route) - 1):
                nd = route[i]
                n1 = route[i - 1]
                n2 = route[i + 1]
                values[nd] = self.times.get(n1, nd) + self.times.get(nd, n2) - self.times.get(n1, n2)
            values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))

        return set(remove)
    
    
    def routeRemoval(self):
        # geometric parameter
        rho = 0.25

        # compute distance
        distance = {}
        for i in range(len(self.routes)):
            route = self.routes[i]
            if len(route) < 3:
                continue

            # compute distance for each node
            total = 0
            for j in range(1, len(route) - 1):
                node = route[j]

                # find min value
                center = float('inf')
                for k in range(len(self.routes)):
                    if k == i:
                        continue
                    tmp = self.routes[k]
                    value = sum(self.times.get(tmp[l], node) for l in range(1, len(tmp) - 1)) / len(tmp)
                    if value < center:
                        center = value
                total += center
            distance[i] = total

        if not distance:
            return set()

        # sort values
        sorted_map = dict(sorted(distance.items(), key=lambda item: item[1]))

        # find index
        index = next(iter(sorted_map))
        for ind in sorted_map:
            if self.rand.random() < rho:
                index = ind
                break
        remove = set(self.routes[index])
        remove.discard(self.depot)

        # remove all nodes
        self.routes[index] = [self.depot, self.depot]
        self.updateRoute(index)

        return remove

    def windowRemoval(self):
        # nodes
        nodes = list(self.param.getNodes())
        nodes.remove(self.depot)

        # parameters
        p = 6
        total = len(nodes)
        limit = round(total * 0.4)
        q = 4 + self.rand.randint(0, min(97, limit))

        # compute diff with window
        values = {}
        for i in range(len(self.routes)):
            route = self.routes[i]
            for j in range(1, len(route) - 1):
                node = route[j]
                vector = self.routesTimes[i][j]
                value = vector[1] - vector[0]
                if value > 0:
                    values[node] = value

        # sort values
        diff = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))

        # list of nodes to remove
        remove = set()
        while len(remove) < q and len(remove) < len(diff):
            # sample position
            y = self.rand.random()
            pos = round(len(diff) * (y ** p))
            pos = max(pos - 1, 0)
            tmp = list(diff.keys())
            remove.add(tmp[pos])

        # remove all nodes in the list
        for nd in remove:
            # find route
            route = []
            ind = -1
            for i in range(len(self.routes)):
                tmp = list(self.routes[i])
                if nd in tmp:
                    tmp.remove(nd)
                    route = list(tmp)
                    ind = i
                    break

            # set route
            self.routes[ind] = route

        self.updateSolution()
        return remove



    def timeRadialRuin(self, div):
        # pick a request at random
        nodes = list(self.param.getNodes())
        nodes.remove(self.depot)
        index = self.rand.randint(0, len(nodes) - 1)
        node = nodes[index]

        # find route
        route = []
        for r in self.routes:
            if node in r:
                route = r
                break
        if not route:
            return set()

        # compute time
        last = route[0]
        time = self.service[last]
        for nd in route:
            # consider the arrival time to check time window
            time += self.times.get(last, nd)

            # wait if too early
            if self.windows[nd][0] > time:
                time = self.windows[nd][0]

            # compute time up to node
            if node == nd:
                break

            # finally sum service time
            time += self.service[nd]
            last = nd

        # threshold
        limit = self.windows[self.depot][1] / div
        limits = [time - limit, time + limit]

        # find all nodes within the proximity
        nodes_to_remove = {node}
        map_routes = {}
        for i, r in enumerate(self.routes):
            map_routes[i] = []
            # compute time
            last = r[0]
            time = self.service[last]
            for j in range(1, len(r) - 1):
                nd = r[j]
                time += self.times.get(last, nd)

                # wait if too early
                if self.windows[nd][0] > time:
                    time = self.windows[nd][0]

                # check time
                if limits[0] <= time <= limits[1]:
                    nodes_to_remove.add(nd)
                    map_routes[i].append(nd)
                if time > limits[1]:
                    break

                # finally sum service time
                time += self.service[nd]
                last = nd

        # remove all nodes in the list
        for ind, nodes_list in map_routes.items():
            route = self.routes[ind]
            for nd in nodes_list:
                route.remove(nd)
            self.routes[ind] = route

        return nodes_to_remove


    def distanceRadialRuin(self, div):
        # pick a request at random
        nodes = list(self.param.getNodes())
        nodes.remove(self.depot)
        index = self.rand.randint(0, len(nodes) - 1)
        node = nodes[index]

        # find route
        route = []
        for r in self.routes:
            if node in r:
                route = r
                break
        if not route:
            return set()

        # compute threshold
        max_dist = 0
        nodes.append(self.depot)
        for n1 in nodes:
            for n2 in nodes:
                dist = self.times.get(n1, n2)
                if dist > max_dist:
                    max_dist = dist

        # find nodes
        nodes_to_remove = {node}
        limit = max_dist / div
        for r in self.routes:
            for j in range(1, len(r) - 1):
                nd = r[j]
                dist = self.times.get(node, nd)
                if dist < limit:
                    nodes_to_remove.add(nd)

        # remove nodes
        for nd in nodes_to_remove:
            for r in self.routes:
                if nd in r:
                    r.remove(nd)

        return nodes_to_remove






    def greedyHeuristic(self, nodesToInsert, noise):
        # noise interval
        maxValue = -1
        nodes = self.param.getNodes()
        for n1 in nodes:
            for n2 in nodes:
                value = self.times.get(n1, n2)
                if value > maxValue:
                    maxValue = value
        plus = maxValue * 0.025

        # tables for cost and locations
        cost = {}
        locations = {}

        # feasible routes indexes
        indexes = [i for i in range(len(self.routes)) if self.feasibleRoutes[i]]

        # iterate until list is empty
        while nodesToInsert:
            # find the best place
            minValue = float('inf')
            insertNode = -1
            insertLocation = -1

            # check previous values
            for node in nodesToInsert:
                if node in cost:
                    row = cost[node]
                    for index, value in row.items():
                        if minValue > value:
                            minValue = value
                            insertNode = node
                            insertLocation = index

            # check modified routes
            for i in indexes[:]:
                route = self.routes[i]
                indexes.remove(i)

                for node in nodesToInsert:
                    # check if feasible
                    if node in cost and i in cost[node] and cost[node][i] >= self.constFeas:
                        continue

                    # find the best position for this node
                    bestInsertValue = float('inf')
                    for j in range(len(route) - 1):
                        # insert in position j + 1
                        value = self.times.get(route[j], node) + self.times.get(node, route[j + 1])

                        # apply noise if needed
                        if noise:
                            unif = plus - self.rand.random() * 2 * plus
                            value = max(value + unif, 0)

                        # create new route
                        tmpRoute = route[:j + 1] + [node] + route[j + 1:]

                        # check value and feasibility
                        if bestInsertValue > value and self.checkFeasible(tmpRoute):
                            bestInsertValue = value
                            if node not in cost:
                                cost[node] = {}
                            cost[node][i] = value
                            locations[(node, i)] = j + 1

                            # if min, update insertNode
                            if value < minValue:
                                minValue = value
                                insertNode = node
                                insertLocation = i

                    # add large value when infeasible
                    if bestInsertValue > self.constFeas:
                        if node not in cost:
                            cost[node] = {}
                        cost[node][i] = self.constFeas

            # insert node if found a feasible option
            if minValue < self.constFeas:
                pos = locations[(insertNode, insertLocation)]
                tmpRoute = self.routes[insertLocation][:pos] + [insertNode] + self.routes[insertLocation][pos:]
                self.routes[insertLocation] = tmpRoute
                self.updateRoute(insertLocation)

                # remove node from list
                nodesToInsert.remove(insertNode)
                if insertNode in cost:
                    del cost[insertNode]
                    keysToDelete = [key for key in locations if key[0] == insertNode]
                    for key in keysToDelete:
                        del locations[key]

                # check this route again
                indexes.append(insertLocation)
                keysToDelete = [key for key in cost if insertLocation in cost[key]]
                for key in keysToDelete:
                    del cost[key][insertLocation]

            else:
                # create empty route for a node
                node = nodesToInsert.pop()
                tmpRoute = [self.depot, node, self.depot]
                self.routes.append(tmpRoute)
                newIndex = len(self.routes) - 1
                self.updateRoute(newIndex)

                # check this route again
                indexes.append(newIndex)

        self.updateSolution()

    def regretHeuristic(self, lenParam, nodesToInsert, noise):
        # number of routes
        if len(self.routes) < 2:
            self.greedyHeuristic(nodesToInsert, noise)
            return

        # check lenParam
        if lenParam < 2:
            return

        # noise interval
        maxDist = -1
        nodes = self.param.getNodes()
        for n1 in nodes:
            for n2 in nodes:
                value = self.times.get(n1, n2)
                if value > maxDist:
                    maxDist = value
        plus = maxDist * 0.025

        # tables for cost and locations
        cost = {}
        locations = {}

        # feasible routes indexes
        indexes = [i for i in range(len(self.routes)) if self.feasibleRoutes[i]]

        # iterate until list is empty
        while nodesToInsert:
            # check modified routes
            for i in indexes[:]:
                route = self.routes[i]
                indexes.remove(i)

                for node in nodesToInsert:
                    # check if feasible
                    if node in cost and i in cost[node] and cost[node][i] >= self.constFeas:
                        continue

                    # find the best position for this node
                    bestInsertValue = float('inf')
                    for j in range(len(route) - 1):
                        # insert in position j + 1
                        value = self.times.get(route[j], node) + self.times.get(node, route[j + 1])

                        # apply noise if needed
                        if noise:
                            unif = plus - self.rand.random() * 2 * plus
                            value = max(value + unif, 0)

                        # create new route
                        tmpRoute = route[:j + 1] + [node] + route[j + 1:]

                        # check value and feasibility
                        if bestInsertValue > value and self.checkFeasible(tmpRoute):
                            bestInsertValue = value
                            if node not in cost:
                                cost[node] = {}
                            cost[node][i] = value
                            locations[(node, i)] = j + 1

                    # add large value when infeasible
                    if node not in cost:
                        cost[node] = {}
                    cost[node][i] = self.constFeas

            # find the best place
            maxRegret = -float('inf')
            insertNode = -1
            insertLocation = -1

            # check previous values
            for node in cost:
                row = sorted(cost[node].items(), key=lambda x: (x[1], x[0]))

                # compute regret
                bestValue = self.constFeas
                regretValue = 0
                cnt = 0
                loc = -1
                for index, value in row:
                    if cnt == 0:
                        bestValue = value
                        loc = index
                    else:
                        regretValue += value - bestValue

                    cnt += 1
                    if cnt == lenParam:
                        break

                # keep the node with the max regret
                if maxRegret < regretValue and bestValue < self.constFeas:
                    maxRegret = regretValue
                    insertNode = node
                    insertLocation = loc

            # insert node if found a feasible option
            if insertNode != -1:
                pos = locations[(insertNode, insertLocation)]
                tmpRoute = self.routes[insertLocation][:pos] + [insertNode] + self.routes[insertLocation][pos:]
                self.routes[insertLocation] = tmpRoute
                self.updateRoute(insertLocation)

                # remove node from list
                nodesToInsert.remove(insertNode)
                if insertNode in cost:
                    del cost[insertNode]
                    keysToDelete = [key for key in locations if key[0] == insertNode]
                    for key in keysToDelete:
                        del locations[key]

                # check this route again
                indexes.append(insertLocation)
                keysToDelete = [key for key in cost if insertLocation in cost[key]]
                for key in keysToDelete:
                    del cost[key][insertLocation]

            else:
                # create empty route for a node
                node = nodesToInsert.pop()
                tmpRoute = [self.depot, node, self.depot]
                self.routes.append(tmpRoute)
                newIndex = len(self.routes) - 1
                self.updateRoute(newIndex)

                # check this route again
                indexes.append(newIndex)

        self.updateSolution()
         
        
    def regretHeuristicAll(self, nodesToInsert, noise):
        self.regretHeuristic(len(self.routes), nodesToInsert, noise)

    ############## PYVRP Operators ###############
    def _to_vrp(self):
        pyVRPRoutes = [ ]
        
        for route in self.routes: 
            r = [int(r) for r in route if r != 0]
            if r:
                pyVRPRoutes.append(r)
        return VRPSolution(self.param.pyVRPData, pyVRPRoutes)

    def _run_route_op(self, op_cls):
        vrp = self._to_vrp()
        rng = RandomNumberGenerator(seed= 5430)
        neighbours = compute_neighbours(self.param.pyVRPData)
        ls = LocalSearch(self.param.pyVRPData, rng, neighbours)
        ls.add_route_operator(op_cls(self.param.pyVRPData))
        penaltyCalc = CostEvaluator(load_penalties =[self.constFeas], 
                                         tw_penalty = self.constFeas, 
                                         dist_penalty=self.constFeas)
        vrp = ls.intensify(vrp, penaltyCalc)
        self.setWithRoutes([[0] + r.visits() +[0] for r in vrp.routes()])
        return self

    def _run_node_op(self, op_cls):
        vrp = self._to_vrp()
        rng = RandomNumberGenerator(seed= 5430 )
        neighbours = compute_neighbours(self.param.pyVRPData)
        ls = LocalSearch(self.param.pyVRPData, rng, neighbours)
        ls.add_node_operator(op_cls(self.param.pyVRPData))
        penaltyCalc = CostEvaluator(load_penalties =[self.constFeas], 
                                         tw_penalty = self.constFeas, 
                                         dist_penalty=self.constFeas)
        vrp = ls.search(vrp, penaltyCalc)
        self.setWithRoutes([[0] + r.visits() +[0] for r in vrp.routes()])
        return self

    def exchange10(self):
        return self._run_node_op(Exchange10)

    def exchange20(self):
        return self._run_node_op(Exchange20)

    def exchange30(self):
        return self._run_node_op(Exchange30)

    def exchange11(self):
        return self._run_node_op(Exchange11)

    def exchange21(self):
        return self._run_node_op(Exchange21)

    def exchange31(self):
        return self._run_node_op(Exchange31)

    def exchange22(self):
        return self._run_node_op(Exchange22)

    def exchange32(self):
        return self._run_node_op(Exchange32)

    def exchange33(self):
        return self._run_node_op(Exchange33)

    # def relocate_with_depot(self):
    #     return self._run_node_op(RelocateWithDepot)

    def swap_tails(self):
        return self._run_node_op(SwapTails)

    def swap_routes(self):
        return self._run_route_op(SwapRoutes)

    def swap_star(self):
        return self._run_route_op(SwapStar)

    ############## END of PYVRP Operators ###############

    def nLowLevel(self):
        return self.nlowlevel

    def applyLowLevel(self, index):
        if index == 0:
            self.searchShift()
        elif index == 1:
            self.searchInterchangeAll()
        elif index == 2:
            self.searchOpt2All()
        elif index == 3:
            self.crossExchangeAll(7)
        elif index == 4:
            self.search2OptInterAll()
        elif index == 5:
            self.pathRelocationAll()
        elif index == 6:
            self.orOpt()
        elif index == 7:
            nodesToInsert = self.timeRadialRuin(15.0)
            self.regretHeuristic(2, nodesToInsert, False)
        elif index == 8:
            nodesToInsert = self.distanceRadialRuin(15.0)
            self.regretHeuristic(2, nodesToInsert, False)
        elif index == 9:
            nodesToInsert = self.timeRadialRuin(20.0)
            self.regretHeuristic(3, nodesToInsert, False)
        elif index == 10:
            nodesToInsert = self.distanceRadialRuin(20.0)
            self.regretHeuristic(3, nodesToInsert, False)
        elif index == 11:
            nodesToInsert = self.shawRemoval()
            self.regretHeuristic(3, nodesToInsert, False)
        elif index == 12:
            nodesToInsert = self.shawRemoval()
            self.regretHeuristic(3, nodesToInsert, True)
        elif index == 13:
            nodesToInsert = self.shawRemoval()
            self.regretHeuristic(4, nodesToInsert, False)
        elif index == 14:
            nodesToInsert = self.shawRemoval()
            self.regretHeuristic(4, nodesToInsert, True)
        elif index == 15:
            nodesToInsert = self.shawRemoval()
            self.regretHeuristicAll(nodesToInsert, False)
        elif index == 16:
            nodesToInsert = self.shawRemoval()
            self.greedyHeuristic(nodesToInsert, False)
        elif index == 17:
            nodesToInsert = self.randomRemoval()
            self.regretHeuristic(4, nodesToInsert, False)
        elif index == 18:
            nodesToInsert = self.randomRemoval()
            self.regretHeuristic(4, nodesToInsert, True)
        elif index == 19:
            nodesToInsert = self.randomRemoval()
            self.greedyHeuristic(nodesToInsert, False)
        elif index == 20:
            nodesToInsert = self.randomRemoval()
            self.regretHeuristicAll(nodesToInsert, False)
        elif index == 21:
            nodesToInsert = self.randomRemoval()
            self.regretHeuristicAll(nodesToInsert, True)
        elif index == 22:
            nodesToInsert = self.worstRemoval()
            self.regretHeuristic(4, nodesToInsert, False)
        elif index == 23:
            nodesToInsert = self.worstRemoval()
            self.regretHeuristic(4, nodesToInsert, True)
        elif index == 24:
            nodesToInsert = self.windowRemoval()
            self.regretHeuristic(3, nodesToInsert, False)
        elif index == 25:
            nodesToInsert = self.routeRemoval()
            self.regretHeuristicAll(nodesToInsert, True)
        elif index == 26:
            nodesToInsert = self.routeRemoval()
            self.regretHeuristicAll(nodesToInsert, False)



