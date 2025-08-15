import numpy as np
import math
import time
import torch
from torch import nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# implements the initilization of the NFEs and the PFEs
class CyclicEmbeddingLayer(nn.Module):

    # node_dim: dimension of the node original info, e.x., 2 for TSP, 7 for VRP
    # embedding_dim: target dimension of feature embeddings
    # num_nodes: number of actual nodes, excluding (dummy) depots in VRP
    # num_dummy_depots: usually used in VRP scenario
    # NOTE: this should be able to work on instances with fewer nodes, but require same node_dim
    def __init__(
            self,
            node_dim,
            embedding_dim,
            num_nodes,
            num_dummy_depots,
            option=0,
            embedder_uniform_distribution=True 
    ):
        super(CyclicEmbeddingLayer, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.max_num_nodes = num_nodes
        self.max_num_dummy_depots = num_dummy_depots
        self.option = option # new implementations?
        self.construction_time = 0
        self.nfe_time = 0
        self.pfe_time = 0

        start_time = time.perf_counter()
        # node feature is mapped from node_dimensional space to embedding_dimensional space
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

        if embedder_uniform_distribution:
            torch.nn.init.xavier_uniform_(self.embedder.weight.data)

        # Two ways for generalizing CPEs: 1. Use the target size CPE directly (default)
        # resulting basis is a matrix of (num_nodes + num_dummy_depots) rows of embedding_dimensional vectors
        if option == 0:
            self.pattern = self.Cyclic_Positional_Encoding(num_nodes + num_dummy_depots, embedding_dim)
        else:
            self.pattern = self.Cyclic_Positional_Encoding(num_nodes + 1, embedding_dim)
        self.construction_time = time.perf_counter() - start_time

    def init_parameters(self):
        # this function can be removed?
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def basesin(self, x, T, fai=0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def basecos(self, x, T, fai=0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    # implements the CPE
    def Cyclic_Positional_Encoding(self, n_position, emb_dim, mean_pooling=True, target_size=None):
        #  pattern is [dummy depot position embedding, real node position embedding]
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype='int')
        x = np.zeros((n_position, emb_dim), dtype=np.float32)

        for i in range(emb_dim):
            Td = Td_set[i // 3 * 3 + 1] if (i // 3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[
                    np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # for generalization (way 2): reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        if target_size is not None:
            pattern = pattern[np.ceil(np.linspace(0, n_position - 1, target_size))]
            pattern_sum = torch.zeros_like(pattern)
            n_position = target_size

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)

        return pattern

    # get index (for permuatation of pattern) according to the solution
    def pattern_index(self, solution, num_real_nodes, num_depots=0):
        seq_length = num_real_nodes + num_depots

        # visited_time is the sequence of timestamp of each node
        # TODO: we could probably get rid of this visited_time and rec representation
        #       but create index from solution sequence directly

        visited_time = torch.zeros(seq_length, device=solution.device)

        if self.option > 0:
            # number of used dummy depots is k
            number_routes = solution[-1][1].item()
            assert number_routes <= self.max_num_dummy_depots
            index = torch.zeros(seq_length, dtype=int, device=solution.device)
            real_dummy_depots = num_depots - number_routes
            k = real_dummy_depots
            for i in range(num_real_nodes):
                index[k] = solution[i][0] + 1
                k = k + 1
                if solution[i][1] > 0:
                    index[k] = 0
                    k = k + 1
            assert k == seq_length
            return index, visited_time

        # visited_time is the sequence of timestamp of each node
        # TODO: we could probably get rid of this visited_time and rec representation
        #       but create index from solution sequence directly

        visited_time = torch.zeros(seq_length, device=solution.device)

        # need to obtain the edge representation:
        # if rec[i] = j, it means the node i is connected to node j, i.e., edge i-j is in the solution
        rec = torch.zeros(seq_length, dtype=int)
        pre = 0

        if num_depots == 0:  # tsp: just same as referrence
            # solutions is supposed to have arrays of num_nodes index while sequence[-1] goes to sequence[0]
            #  obtain the edge representation:
            for j in range(seq_length - 1):
                rec[solution[j]] = solution[j + 1]
            rec[solution[-1]] = solution[0]

            # obtain the index from edge representation:
            for i in range(seq_length):
                visited_time[rec[pre]] = i + 1
                pre = rec[pre]
            index = (visited_time % seq_length).int()  # .long().unsqueeze(-1).expand(seq_length, self.embedding_dim)
        else:  # vrp
            # solutions is supposed to have arrays of num_nodes index while last column indicate which depot it goes to
            # note: flag == 0 means not going to depot, flag = i means on route i-1, or going to depot i-1
            #  obtain the edge representation:

            # intialize all dummy depots as singleton
            # the first element in solution sequence must be from a depot
            # the last element in the solution sequence must go to a depot
            for k in range(num_depots):
                rec[k] = -1
            source = solution[0][0] + num_depots
            for j in range(num_real_nodes):
                if solution[j][1] <= 0:  # not going to depot, real node
                    rec[solution[j][0] + num_depots] = solution[j + 1][0] + num_depots
                else:  # going to depot
                    rec[solution[j][0] + num_depots] = solution[j][1] - 1
                    if (j < num_real_nodes - 1):  # not finished
                        rec[solution[j][1] - 1] = solution[j + 1][0] + num_depots
            rec[solution[num_real_nodes - 1][1] - 1] = source
            open_depots = list(np.where(rec[range(num_depots)] < 0))[0]
            if len(open_depots) != 0:  # some depots are not linked yet
                # link the above route by the fist open dummy depot
                rec[solution[num_real_nodes - 1][1] - 1] = open_depots[0]
                num_open_depot = len(open_depots)
                for k in range(num_open_depot - 1):
                    rec[open_depots[k]] = open_depots[k + 1]
                rec[open_depots[num_open_depot - 1]] = source

                # obtain the index from edge representation:
            for i in range(seq_length):
                visited_time[rec[pre]] = i + 1
                pre = rec[pre]

            index = np.argsort(
                visited_time % seq_length).int()  # .long().unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)

        return index, visited_time

    # solution: the solution of a particular instance
    # num_depots: the total number of dummy depots
    def position_encoding(self, solution, num_depots=0):
        # expand for every batch with trimed patterns: dummpy-depots pattern subset + node pattern subset
        num_real_nodes = solution.size(dim=0)
        if self.option > 0:
            # get index according to the solutions
            index, visited_time = self.pattern_index(solution, num_real_nodes, num_depots)
            pfe = self.pattern.clone()[index]
            return pfe
        
        
        pattern_index = list(range(num_depots)) + list(
            range(self.max_num_dummy_depots, self.max_num_dummy_depots + num_real_nodes))
        CPE_embeddings = self.pattern.clone().to(solution.device)[pattern_index, :]

        # get index according to the solutions
        index, visited_time = self.pattern_index(solution, num_real_nodes, num_depots)
        pfe = CPE_embeddings[index]
        return pfe

    # x: input of assembled (original) node info, dimension = M * self.node_dim, while M = m1 + m2 + .. + mk, each slice includes the dummy depots involved in the instance
    # solution: input of assembled (original) solution, dimension = N * 2,  while N = n1 + n2 + .. + nk, each slice includes the first column as an array of "real" nodes and
    # second column as a "flagging" column (taking value k>0 means goes to depot k-1 in route k-1; 0 means NO), note that the dummy depots involved in the instance are EXCLUDED
    # node_info_sizes is the array of node info data sizes of each instance
    # solution_sizes is the array of solution info data sizes of each instance
    def forward(self, node_info_assembled, solutions_assembled, node_info_sizes=None, solution_sizes=None):
        # positional feature embedding, dim = node_info_assembled.dim(0) * self.embedding_dim
        if node_info_sizes == None:
            PFEs = self.position_encoding(solutions_assembled)
        else:
            assert solution_sizes != None
            assert node_info_sizes.size() == solution_sizes.size()
            num_blocks = node_info_sizes.size(dim=0)

            PFEs = torch.empty(0, self.embedding_dim)
            pos = 0
            for i in range(num_blocks):
                num_depots = node_info_sizes[i].item() - solution_sizes[i].item()
                pfe = self.position_encoding(solutions_assembled[range(pos, pos + solution_sizes[i].item()), :],
                                             num_depots)
                PFEs = torch.cat([PFEs, pfe], dim=0)
                pos += solution_sizes[i].item()

        # node feature embedding, dim = node_info_assembled.dim(0) * self.embedding_dim
        NFEs = self.embedder(node_info_assembled)

        return NFEs, PFEs

    def forward_fe(self, node_info_assembled : torch.FloatTensor) -> torch.FloatTensor :
        # node feature embedding, dim = node_info_assembled.dim(0) * self.embedding_dim
        start_time = time.perf_counter()
        NFEs = self.embedder(node_info_assembled)
        self.nfe_time += time.perf_counter() - start_time
        return NFEs
    def forward_pe(self, solutions_assembled : torch.FloatTensor, node_info_sizes : torch.LongTensor, solution_sizes : torch.LongTensor) -> torch.FloatTensor :
        assert node_info_sizes.size() == solution_sizes.size()
        start_time = time.perf_counter()
        num_blocks = node_info_sizes.size(dim=0)

        PFEs = torch.empty(0, self.embedding_dim)
        pos = 0
        for i in range(num_blocks):
            num_depots = node_info_sizes[i].item() - solution_sizes[i].item()
            pfe = self.position_encoding(solutions_assembled[range(pos, pos + solution_sizes[i].item()), :],
                                         num_depots)
            PFEs = torch.cat([PFEs, pfe], dim=0)
            pos += solution_sizes[i].item()
        self.nfe_time += time.perf_counter() - start_time
        return PFEs

    def display_time(self):
        print("Feature Embedding Profiler: construction {:.4f}, nfe {:.4f}, pfe {:.4f}".format(self.construction_time, self.nfe_time, self.nfe_time))
