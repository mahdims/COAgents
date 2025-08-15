import typing

import numpy as np
import torch as th
import torch_geometric as tg

from collections import OrderedDict


""" Taken from posenc_stats.py @ https://github.com/rampasek/GraphGPS

Ladislav Rampasek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu and Guy Wolf
"Recipe for a General, Powerful, Scalable Graph Transformer",
 36th Conference on Neural Information Processing Systems (NeurIPS 2022).

"""
# def get_rw_landing_probs (ksteps : int, edge_index : th.Tensor, edge_weight : th.Tensor = None,
#                          num_nodes : int = 0, space_dim : int = 0) -> th.Tensor :
#     """Compute Random Walk landing probabilities for given list of K steps.
#
#     Args:
#         ksteps: List of k-steps for which to compute the RW landings
#         edge_index: PG sparse representation of the graph
#         edge_weight: (optional) Edge weights
#         num_nodes: (optional) Number of nodes in the graph
#         space_dim: (optional) Estimated dimensionality of the space. Used to
#             correct the random-walk diagonal by a factor `k^(space_dim/2)`.
#             In euclidean space, this correction means that the height of
#             the gaussian distribution stays almost constant across the number of
#             steps, if `space_dim` is the dimension of the euclidean space.
#
#     Returns:
#         2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
#     """
#
#     # ToDo: rewrite with numpy
#     if type(edge_index) == np.ndarray :
#         edge_index = th.from_numpy(edge_index).to(th.long)
#
#     device = edge_index.device
#     if edge_weight is None :
#         edge_weight = th.ones(edge_index.size(1), device=device)
#     source, dest = edge_index[0], edge_index[1]
#     deg = tg.utils.scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
#     deg_inv = deg.pow(-1.)
#     deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
#
#     if edge_index.numel() == 0 :
#         P = th.zeros((1, num_nodes, num_nodes), dtypy=th.float32, devie=device)
#     else :
#         # P = D^-1 * A
#         P = th.diag(deg_inv) @ tg.utils.to_dense_adj(edge_index, max_num_nodes=num_nodes)
#     rws = []
#     # Efficient way if ksteps are a consecutive sequence (most of the time the case)
#     Pk = P.clone().detach()
#     for k in range(ksteps) :
#          rws.append(th.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
#          Pk = th.matmul(Pk, P)
#     rw_landing = th.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)
#     return rw_landing

# The random walk graph positional embedding
def get_rw_landing_probs (ksteps : int, e : np.ndarray, edges_weight : typing.Optional[ np.ndarray ] = None, num_nodes : int = 0, space_dim : int = 0) -> np.ndarray :

    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """

    deg = np.zeros(shape=(num_nodes, ), dtype=np.float32)
    if edges_weight is None :
        np.add.at(deg, e[:, 0], 1.)
        np.add.at(deg, e[:, 1], 1.)
    else                    :
        np.add.at(deg, e[:, 0], edges_weight)
        np.add.at(deg, e[:, 1], edges_weight)
    deg_inv = 1. / deg
    deg_inv[deg_inv == float('inf')] = 0.
    if not e.size :
         p = np.zeros((1, num_nodes, num_nodes), dtype=th.float32)
    else :
         # P = D^-1 * A
         A = np.zeros(shape=(num_nodes, num_nodes,), dtype=np.float32)
         A[e[:,0], e[:,1]] = 1. ; A[e[:,1], e[:,0]] = 1.
         p = np.matmul(np.diag(deg_inv), A)
    rws = []
    # Efficient way if ksteps are a consecutive sequence (most of the time the case)
    p_k = p.copy()
    for k in range(ksteps) :
        rws.append((np.diag(p_k) * (k ** (space_dim / 2.)))[:,np.newaxis])
        p_k = np.matmul(p_k, p)
    rw_landing = np.concatenate(rws, axis=1)  # (Num nodes) x (K steps)
    return rw_landing


# Implements graph embedding
class GraphEmbeddingLayer(th.nn.Module) :
    def __init__(self,
                 i_dim  : int ,   # input dimension
                 p_dim  : int ,   # position embedding dimension
                 o_dim  : int ,   # output dimension
                 nedges_types : int, # maximum of edges types
                 ez_dim : int     # number of edge embeddings
                 ) :
        super(GraphEmbeddingLayer, self).__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim
        self.p_dim = p_dim
        self.nedges_types = nedges_types
        self.ez_dim = ez_dim

        # The node embedder
        self.gembedding_layer  = th.nn.Sequential(OrderedDict([
            ('linear1', th.nn.Linear(i_dim + p_dim, 2 * o_dim, bias=False, device=th.device('cpu'), dtype=th.float32)),
            ('relu1',   th.nn.ReLU()),
            ('linear2', th.nn.Linear(2 * o_dim,         o_dim, bias=True,  device=th.device('cpu'), dtype=th.float32)),
            ('relu2',   th.nn.ReLU())
        ]))
        th.nn.init.xavier_uniform_(self.gembedding_layer[0].weight.data)
        th.nn.init.xavier_uniform_(self.gembedding_layer[2].weight.data) ; th.nn.init.zeros_(self.gembedding_layer[2].bias)
        # The edge embedder
        #self.eembedding_layer = th.nn.Embedding(nedges_types, ez_dim, max_norm=1.0)
        self.eembedding_layer = th.nn.Embedding(nedges_types, ez_dim)

    # The forward
    def forward (self,
                 i : th.FloatTensor, # Input feature embedding tensor
                 p : th.FloatTensor, # Input position embeding tensor
                 e  : th.LongTensor,  # edges ids in the solution sample
                 ) -> tuple[ th.FloatTensor, th.FloatTensor ] : # graph nodes embeddings and graph edges embeddings
        # Gradient enabled part
        assert i.shape[0] == p.shape[0] , "Sizes of features and position emebddings doesn't match!"
        f = th.cat((i, p), dim=1).requires_grad_(False)
        g_embedding = self.gembedding_layer(f)
        e_embedding = self.forward_edges(e[:, 2])
        return (g_embedding, e_embedding)


    # This routine computes the embedding of edges (moves)
    def forward_edges(self, e_id : th.LongTensor) -> th.FloatTensor :
        return self.eembedding_layer(e_id)
