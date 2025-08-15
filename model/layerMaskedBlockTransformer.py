
# This module is fast but limited to the same size vrp instances

import typing
import torch as th
import math
from collections import OrderedDict


def simple_normalized_aggregation(n : th.LongTensor, l : int, z_nheads : int, z : th.FloatTensor) -> th.FloatTensor :
    assert (len(z.shape) == 2) and (not z.size(1) % z_nheads) , "Inconsistent shapes and heads number"
    z = th.nn.functional.normalize(z.view(z.size(0), z_nheads, -1), p=2, dim=-1, eps=1e-12).view(z.size(0), -1)
    z_new = th.zeros(n.sum(), z.shape[-1], dtype=z.dtype, device=z.device, requires_grad=True)
    z_new = z_new.index_add(0, th.arange(n.sum(), dtype=th.long, device=z.device).repeat(l), z, alpha=1./float(l))
    return z_new

class MaskedBlockTransformer(th.nn.Module) :
    def __init__(self, u_dim : int, v_dim : int, z_dim : int, nheads : int, z_nheads : int, dropout : float = 0.1) -> None :
        super(MaskedBlockTransformer, self).__init__()
        self.u_dim  = u_dim
        self.v_dim  = v_dim
        self.z_dim  = z_dim
        self.nheads = nheads
        self.z_nheads = z_nheads
        assert not v_dim % self.nheads , "The dimension of v tensor should be divisible by the number of heads."
        assert not z_dim % self.z_nheads , "The dimension of z tensor should be divisible by the number of z-heads."
        self.inv_sqrt_z_dim = 1. / math.sqrt(float(z_dim))

        self.Z = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear((self.u_dim + self.v_dim), 2 * (self.u_dim + self.v_dim), bias=True, dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * (self.u_dim + self.v_dim),  self.z_dim,               bias=True, dtype=th.float32))
        ]))

        self.Q = th.nn.Parameter(th.empty(self.nheads, self.z_dim, self.z_dim, dtype=th.float32))
        self.K = th.nn.Parameter(th.empty(self.nheads, self.z_dim, self.z_dim, dtype=th.float32))
        self.V = th.nn.Parameter(th.empty(self.nheads, self.v_dim // self.nheads, self.v_dim // self.nheads, dtype=th.float32))

        self.F = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear(v_dim, 2 * (v_dim + v_dim), bias=True, dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * (v_dim + v_dim), v_dim, bias=True, dtype=th.float32))
        ]))
        th.nn.init.xavier_uniform_(self.Z[0].weight.data) ; th.nn.init.zeros_(self.Z[0].bias)
        th.nn.init.xavier_uniform_(self.Z[2].weight.data) ; th.nn.init.zeros_(self.Z[2].bias)
        th.nn.init.xavier_uniform_(self.V)
        th.nn.init.xavier_uniform_(self.Q)
        th.nn.init.xavier_uniform_(self.K)

        th.nn.init.xavier_uniform_(self.F[0].weight.data) ; th.nn.init.zeros_(self.F[0].bias)
        th.nn.init.xavier_uniform_(self.F[2].weight.data) ; th.nn.init.zeros_(self.F[2].bias)

        self.bn_node_vo = th.nn.BatchNorm1d(v_dim)
        self.bn_node_v  = th.nn.BatchNorm1d(v_dim)

        self.dropout = dropout

        self.grad_tensors = [] # For debugging



    def forward(self,
                n : th.LongTensor,
                l : int,
                u : th.FloatTensor,
                v : th.FloatTensor,
                debug_layers : typing.Optional[typing.List[typing.Tuple[str, th.Tensor ]]] = None,
                ) -> th.FloatTensor:
        assert not v.size(-1) % self.nheads , "The dimension of v tensor should be divisible by the number of heads."
        # Debug
        #debug_layers.append(("u", u)) ; u.retain_grad()
        #debug_layers.append(("v", v)) ; v.retain_grad()

        # Construct global embedding z using simple multi-heads attention
        z = self.Z.forward(th.concatenate((th.repeat_interleave(u, l, dim=0), v), dim=1))
        z = simple_normalized_aggregation(n, l, self.z_nheads, z)
        #Debug
        #debug_layers.append(("z", z)) ; z.retain_grad()

        # Construct query and key matrices
        q = th.bmm(z.unsqueeze(0).expand(self.nheads, -1, -1), self.Q) ; k = th.bmm(z.unsqueeze(0).expand(self.nheads, -1, -1), self.K)
        #Debug
        #debug_layers.append(("q", q)) ; q.retain_grad()
        #debug_layers.append(("k", k)) ; k.retain_grad()

        # Compute attention mask
        block_indices  = th.repeat_interleave(th.arange(len(n), dtype=th.long, device=n.device), n)
        attn_mask = (block_indices.unsqueeze(1) != block_indices.unsqueeze(0))
        attn_mask = attn_mask.unsqueeze(0)  # adds head dimensions
        # Compute masked attention matrix
        attn_matrix = th.bmm(q, k.transpose(1, 2))
        attn_matrix = attn_matrix.masked_fill(attn_mask, float('-inf')) # H x N x N
        # Compute softmax normalization
        attn_matrix = th.nn.functional.softmax(attn_matrix * self.inv_sqrt_z_dim, dim=2)
        # Debug
        #debug_layers.append(("attn_matrix", attn_matrix)) ; attn_matrix.retain_grad()

        # Multiply vi matrix with attention matrix
        # Construct values matrix
        vi = th.bmm(v.view(-1, self.nheads, v.size(-1) // self.nheads).transpose(0, 1), self.V) # (B*l) x F -> H x (B*l) x F
        # Debug
        #debug_layers.append(("vi", vi)) ; vi.retain_grad()

        # Multiply attentions
        #vo = th.einsum('h n m, h m l f -> h n l f', attn_matrix,  vi.view(nheads, nbatch, -1, nfeats)) # requires permtation
        #vo = th.bmm(th.repeat_interleave(attn_matrix, l, dim=0), vi.view(nheads, nbatch, -1, nfeats).transpose(1,2).reshape(-1, nbatch, nfeats)).view(nheads, l, nbatch, nfeats).transpose(1,2) # slow 2x
        vo = th.einsum('h n m, h m l f -> n l h f', attn_matrix, vi.view(self.nheads, -1, l, vi.size(-1))) # B x L x H x F
        vo = vo.reshape(-1, v.size(-1))
        # Debug
        #debug_layers.append(("vo", vo)) ; vo.retain_grad()

        # Dropout
        vo = th.nn.functional.dropout(vo, p=self.dropout, training=self.training)
        # Normalization postprocessing of residual connection
        #vo = self.bn_node_vo(v + vo)

        # Project batch the output
        v_new = self.F.forward(vo)
        # Debug
        #debug_layers.append(("v_new", v_new)) ; v_new.retain_grad()

        # Normalization final postprocessing
        v_new = self.bn_node_v(v_new)

        return v_new


if __name__ == "__main__" :
    u_dim  = 11
    z_dim  = 14
    v_dim  = 6
    nheads = 3
    z_nheads = 2
    assert not v_dim % nheads , "Incorrect v_dim or nheads"
    th.manual_seed(2004)

    # Create the layer
    mbt = MaskedBlockTransformer(u_dim, v_dim, z_dim, nheads, z_nheads)

    # Create data for the unittest
    n = th.tensor([3, 6, 2], dtype=th.long)
    l = 4
    u = th.rand(n.sum(),     u_dim, requires_grad=True)
    v = th.rand(n.sum() * l, v_dim, requires_grad=True)

    # Forward pass
    (u_new, v_new) = mbt.forward(n, l, u, v)

    print("Successfully completed!")


