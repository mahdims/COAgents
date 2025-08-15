from collections import defaultdict

class EfficientTime:
    def __init__(self, times):
        # Stores distances with (node1, node2) as key
        self.times = times
        # Stores the maximum distance for each node
        self.maxTimes = defaultdict(float)
        for (node1, node2), value in self.times.items():
            self.maxTimes[node1] = max(self.maxTimes[node1], value)
            self.maxTimes[node2] = max(self.maxTimes[node2], value)

    def max(self, node):
        return self.maxTimes[node]

    def get(self, n1, n2):
        return self.times.get((n1,n2))
    
    def __getitem__(self, key):
        return self.times.get(key)