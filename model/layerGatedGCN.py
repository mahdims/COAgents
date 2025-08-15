import torch as th
import typing

from collections import OrderedDict

# The gated GCN Layer
#class GatedGCNLayer(tg.conv.MessagePassing):
class GatedGCNLayer(th.nn.Module) :

    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, u_dim : int, v_dim : int, e_dim : int, z_dim : int, dropout : float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.u_dim = u_dim # g_dim - global vector dimension
        self.v_dim = v_dim # h_dim - local node vector dimension
        self.e_dim = e_dim # e_dim - edge embedding dimension
        self.z_dim = z_dim # z_dim - hidden dimension


        self.Z   = th.nn.Linear(u_dim + v_dim, z_dim, bias=True)  # Embedding of global data
        self.H   = th.nn.Linear(v_dim,         z_dim, bias=True)  # Embedding of the local data
        self.K   = th.nn.Linear(z_dim,         z_dim, bias=True)  # Ti Gate transformer in-take
        self.Q   = th.nn.Linear(z_dim,         z_dim, bias=True)  # Tj Gate transformer other
        self.Eij = th.nn.Linear(e_dim,         z_dim, bias=True)  # Edge type embedding forward
        self.Eji = th.nn.Linear(e_dim,         z_dim, bias=True)  # Edge type embedding backward
        self.U = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear(z_dim, 2 * (z_dim + u_dim), bias=True, dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * (z_dim + u_dim), u_dim, bias=True, dtype=th.float32))
        ]))
        self.V = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear(z_dim, 2 * (z_dim + v_dim), bias=True, dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * (z_dim + v_dim), v_dim, bias=True, dtype=th.float32))
        ]))
        th.nn.init.xavier_uniform_(self.Z.weight.data) ; th.nn.init.zeros_(self.Z.bias)
        th.nn.init.xavier_uniform_(self.H.weight.data) ; th.nn.init.zeros_(self.H.bias)
        th.nn.init.xavier_uniform_(self.K.weight.data) ; th.nn.init.zeros_(self.K.bias)
        th.nn.init.xavier_uniform_(self.Q.weight.data) ; th.nn.init.zeros_(self.Q.bias)
        th.nn.init.xavier_uniform_(self.Eij.weight.data) ; th.nn.init.zeros_(self.Eij.bias)
        th.nn.init.xavier_uniform_(self.Eji.weight.data) ; th.nn.init.zeros_(self.Eji.bias)
        th.nn.init.xavier_uniform_(self.U[0].weight.data) ; th.nn.init.zeros_(self.U[0].bias)
        th.nn.init.xavier_uniform_(self.U[2].weight.data) ; th.nn.init.zeros_(self.U[2].bias)
        th.nn.init.xavier_uniform_(self.V[0].weight.data) ; th.nn.init.zeros_(self.V[0].bias)
        th.nn.init.xavier_uniform_(self.V[2].weight.data) ; th.nn.init.zeros_(self.V[2].bias)

        self.bn_node_u = th.nn.BatchNorm1d(u_dim)
        self.bn_node_v = th.nn.BatchNorm1d(v_dim)
        self.bn_node_h = th.nn.BatchNorm1d(z_dim)
        self.dropout = dropout

    # The 1D convolution layer
    @staticmethod
    def conv1D(n : th.LongTensor,
               l : int,
               u : th.FloatTensor,
               v : th.FloatTensor,
               Z : th.nn.Module,
               sc_indx : th.LongTensor,
               ) -> th.FloatTensor:
        s_conv = th.nn.functional.gelu(Z.forward(th.cat((u.repeat_interleave(l, 0), v), dim=1)))
        z = th.zeros(n.sum(), s_conv.shape[1], dtype=v.dtype, device=v.device).index_add(0, sc_indx, s_conv) / float(l)
        return z

    # The forward routine
    def forward(self,
                n  : th.LongTensor,
                l  : int,
                u  : th.FloatTensor,
                v  : th.FloatTensor,
                ez : th.FloatTensor,
                e  : th.LongTensor,
                sc_indx : typing.Union[ th.LongTensor, None ], # Solutiion convolution indices (for speedup)
                ) -> th.FloatTensor :
        """
        n               : [n_nodes]
        l               : [n_nodes * l
        u               : [n_nodes, g_dim ]
        v               : [n_nodes * l, h_dim ]
        ez              : [n_edges, e_dim]
        e               : [2, n_edges]
        """

        """
            The adaptation to structured input of (VRP) nodes
            Compute self-representation of the node as gated sum:
            h_z = H.v
            z = 1/N * sum(relu(G.(g ^ v)))
            Compute the gate:
            gate_ij = tanh(K.z_i + Q.z_j + E.e_ij)
            Compute the aggregation of messages:
            h_z_new = relu(h_zi + sum[j](h_zj * gate_ij)) 
            Compute global representation back:
            u_new = U.(z + 1/N * sum(h_z_new))
            v_new = V.(h_z + h_z_new)
        """

        # Construct global embedding z with 1D convolution inside the solutions
        z = self.conv1D(n, l,  u, v, self.Z, sc_indx=sc_indx)

        # Message passing
        h_z     = self.H.forward(v)
        query_i = self.K.forward(z)
        key_j   = self.Q.forward(z)
        e_ij    = self.Eij.forward(ez)
        e_ji    = self.Eji.forward(ez)
        gate_ij = th.nn.functional.tanh(key_j[e[:, 1]] + query_i[e[:, 0]] + e_ij) # ToDo: sum nodes with the same type of move
        etag_ji = th.nn.functional.tanh(key_j[e[:, 0]] + query_i[e[:, 1]] + e_ji) # ToDo: sum nodes with the same type of move

        # Message passing
        # ToDo: Add normalization by the amount of edges
        h_z_new = th.zeros(h_z.shape[0] // l, l * h_z.shape[-1], dtype=h_z.dtype, device=h_z.device)
        h_z_new.index_add_(0, e[:, 0], etag_ji.repeat(1, l) * h_z.view(-1, l, h_z.shape[-1]).view(-1, l * h_z.shape[-1])[e[:, 1]])
        h_z_new.index_add_(0, e[:, 1], gate_ij.repeat(1, l) * h_z.view(-1, l, h_z.shape[-1]).view(-1, l * h_z.shape[-1])[e[:, 0]])

        # Normalization
        h_z_new = self.bn_node_h(h_z_new.view(-1, h_z.shape[-1]))
        if self.dropout :
            h_z_new = th.nn.functional.dropout(h_z_new, p=self.dropout, training=self.training)

        # 1D de-convolution withing the solutions
        u_new = self.U.forward(z + th.zeros_like(z).index_add(0, sc_indx, h_z_new) / float(l))
        v_new = self.V.forward(h_z + h_z_new)

        # Normalization
        u_new = self.bn_node_u(u_new)
        v_new = self.bn_node_v(v_new)

        return (u_new, v_new)


