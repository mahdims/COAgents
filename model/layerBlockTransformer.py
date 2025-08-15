
import typing
import torch as th

from collections import OrderedDict

# Taken from https://github.com/hyunwoongko/transformer/blob/master/models/layers/
class BlockTransformer(th.nn.Module):
    def __init__(self, u_dim : int, v_dim : int, z_dim : int, nheads : int, dropout : float = 0.1) -> None :
        super(BlockTransformer, self).__init__()
        self.u_dim  = u_dim
        self.v_dim  = v_dim
        self.z_dim  = z_dim
        self.nheads = nheads
        assert ( (z_dim >= self.nheads) and (v_dim >= self.nheads) ) , "Too short dimensions (or too many attention heads)."
        assert ( (not z_dim % self.nheads) and (not v_dim % self.nheads) ) , "The dimension of tensor should be divisible by the number of heads."
        self.inv_sqrt_dim = th.FloatTensor([float(nheads) / float(z_dim),],).sqrt().item()

        #self.Z = th.nn.Linear(self.u_dim + self.v_dim, self.z_dim)
        self.Z = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear((self.u_dim + self.v_dim), 2 * (self.u_dim + self.v_dim), bias=True, dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * (self.u_dim + self.v_dim),  self.z_dim,               bias=True, dtype=th.float32))
        ]))
        self.V = th.nn.Linear(self.v_dim, self.z_dim, bias=True)
        self.Q = th.nn.Linear(self.z_dim, self.z_dim, bias=True)
        self.K = th.nn.Linear(self.z_dim, self.z_dim, bias=True)
        self.W = th.nn.Linear(self.z_dim, self.v_dim, bias=True)
        self.F = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear(v_dim, (v_dim + v_dim), bias=True, dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear((v_dim + v_dim), v_dim, bias=True, dtype=th.float32))
        ]))
        #th.nn.init.xavier_uniform_(self.Z.weight.data) ; th.nn.init.zeros_(self.Z.bias)
        th.nn.init.xavier_uniform_(self.Z[0].weight.data) ; th.nn.init.zeros_(self.Z[0].bias)
        th.nn.init.xavier_uniform_(self.Z[2].weight.data) ; th.nn.init.zeros_(self.Z[2].bias)
        th.nn.init.xavier_uniform_(self.V.weight.data) ; th.nn.init.zeros_(self.V.bias)
        th.nn.init.xavier_uniform_(self.Q.weight.data) ; th.nn.init.zeros_(self.Q.bias)
        th.nn.init.xavier_uniform_(self.K.weight.data) ; th.nn.init.zeros_(self.K.bias)
        th.nn.init.xavier_uniform_(self.W.weight.data) ; th.nn.init.zeros_(self.W.bias)
        th.nn.init.xavier_uniform_(self.F[0].weight.data) ; th.nn.init.zeros_(self.F[0].bias)
        th.nn.init.xavier_uniform_(self.F[2].weight.data) ; th.nn.init.zeros_(self.F[2].bias)

        self.bn_node_vo = th.nn.BatchNorm1d(v_dim)
        self.bn_node_v  = th.nn.BatchNorm1d(v_dim)

        self.dropout = dropout

        self.grad_tensors = [] # For debugging


    # The 1D convolution layer
    @staticmethod
    def conv1D (l : th.LongTensor,
                u : th.FloatTensor,
                v : th.FloatTensor,
                Z : th.nn.Module,
                sc_indx : th.LongTensor,
                ) -> th.FloatTensor:
        s_conv = th.nn.functional.relu(Z.forward(th.cat((u.repeat_interleave(l, 0), v), dim=1)))
        z = th.zeros(l.shape[0], s_conv.shape[1], dtype=v.dtype, device=v.device).index_add(0, sc_indx, s_conv) / l.view(-1, 1).to(dtype=v.dtype)
        return z

    def forward(self,
                n : th.LongTensor,
                s : th.LongTensor,
                l : th.LongTensor,
                u : th.FloatTensor,
                v : th.FloatTensor,
                sc_indx  : th.LongTensor, # sc_indx = th.arange(l.nelement(), dtype=th.long, device=l.device).repeat_interleave(l, 0)
                indx_sa  : typing.Optional[th.LongTensor] = None, # indx_sa = BlockTransformer.compute_block_attention_indices(n, u_dim, nheads)
                indx_vf  : typing.Optional[th.LongTensor] = None, # indx_vf = BlockTransformer.compute_multihead_features_indices_flatten(l, v_dim, nheads)
                indx_vs  : typing.Optional[th.LongTensor] = None, # indx_vs = BlockTransformer.compute_block_features_indices_product(n, l, s, v_dim, nheads)
                indx_vsf : typing.Optional[th.LongTensor] = None, # indx_vsf = BlockTransformer.compute_block_features_indices_flatten(n, s, v_dim, nheads)
                ) -> th.FloatTensor:

        # Construct global embedding z
        # u.retain_grad() ; v.retain_grad()
        # self.grad_tensors.append(("u", u)) ; self.grad_tensors.append(("v", v))
        z = self.conv1D(l, u, v, self.Z, sc_indx=sc_indx)
        # Construct query and key matrices
        q = self.Q.forward(z) ; k = self.K.forward(z) ; vi = self.V.forward(v)

        # Compute attention
        if n.nelement() == 1 : # block-attention
            vo = self.dense_attention(self.z_dim, self.z_dim, self.nheads, n, s, q, k, vi, inv_sqrt_dim=self.inv_sqrt_dim,)
        else                 : # traditional attention, good for 1 sample
            if indx_sa  is None : indx_sa  = BlockTransformer.compute_block_attention_indices(n, self.z_dim, self.nheads)
            if indx_vf  is None : indx_vf  = BlockTransformer.compute_multihead_features_indices_flatten(l, self.z_dim, self.nheads)
            if indx_vs  is None : indx_vs  = BlockTransformer.compute_block_features_indices_product(n, l, s, self.z_dim, self.nheads)
            if indx_vsf is None : indx_vsf = BlockTransformer.compute_block_features_indices_flatten(n, s, self.z_dim, self.nheads)
            vo = self.block_attention(self.z_dim, self.z_dim, self.nheads, n, s, q, k, vi,
                                      indx_sa, indx_vf, indx_vs, indx_vsf, inv_sqrt_dim=self.inv_sqrt_dim, )
        vo = self.W.forward(vo) # Project back into input dimension

        # Dropout
        vo = th.nn.functional.dropout(vo, p=self.dropout, training=self.training)
        # Normalization postprocessing of residual connection
        vo = self.bn_node_vo(v + vo)

        # Project batch the output
        v_new = self.F.forward(vo)

        # Dropout final
        v_new = th.nn.functional.dropout(v_new, p=self.dropout, training=self.training)
        # Normalization final postprocessing
        v_new = self.bn_node_v(v_new)

        return v_new

    # The transformer itself

    # The class-methods for block-transformers
    """
    The (sparse) multihead [sample_head] query block-matrix:
    | s0_0 |                           |
    | s1_0 |                           |
    |      | s0_1 |                    |
    |      | s1_1 |                    |
    |             | s2_0 |             |
    |             | s3_0 |             |
    |             | s4_0 |             |
    |                    | s2_1 |      |
    |                    | s3_1 |      |
    |                    | s4_1 |      |
    |                           | s2_2 |
    |                           | s3_2 |
    |                           | s4_2 |
    Each block of q is n_sample x (n_sample * u_dim / nheads)
    The k matrix is the transpose of q shape.
    """
    # This routine computes tensor indices
    @staticmethod
    def compute_block_attention_indices(n : th.LongTensor, z_dim : int, nheads : int = 1) -> th.LongTensor :
        assert not z_dim % nheads , "The dimension of tensor should be divisible by the number of heads."
        # Dimension per-head
        tz = th.zeros(1, dtype=th.long, device=n.device)
        h_dim = z_dim // nheads
        # Elementary increment of indices withing a block
        offset_j = (z_dim * th.arange(n.nelement(), dtype=th.long, device=n.device)).repeat_interleave(z_dim * n, dim=0)
        indx_j   = offset_j + th.arange(z_dim, dtype=th.long, device=n.device).repeat(n.sum(dtype=th.long)) # col_id = offset_j +
        indx_i   = th.arange(z_dim * n.sum(dtype=th.long), dtype=th.long, device=n.device) # initialize as a plain enumeration
        row_frst = (th.cat((tz, th.cumsum(n, dim=0)), dim=0)[:-1]).repeat_interleave(z_dim * n, dim=0) # The first row of a group
        stride_i = ((indx_i % z_dim) // h_dim) * n.repeat_interleave(z_dim * n)
        indx_i   = nheads * row_frst + stride_i + (indx_i // z_dim - row_frst)
        return th.stack((indx_i, indx_j), dim=0)
    # The sparse attention matrix
    @staticmethod
    def compute_block_attention_matrix(n : th.LongTensor, q : th.FloatTensor, k : th.FloatTensor, indxs : th.LongTensor, z_dim : int, nheads : int = 1) -> th.FloatTensor :
        # Sanity checks
        assert (q.shape == k.shape) , "The input keys and queries tensors should be of the same shape."
        assert ( (len(q.shape) == 2) and (q.shape[1] == z_dim) ) , "Unexpected shape for q and k."
        # Construct sparse matrix
        qs = th.sparse_coo_tensor(indxs, q.flatten(), (nheads * n.sum(dtype=th.long), n.nelement() * z_dim)).coalesce()
        ks = th.sparse_coo_tensor(indxs, k.flatten(), (nheads * n.sum(dtype=th.long), n.nelement() * z_dim)).coalesce().transpose(0, 1)
        # Compute attention
        a_sparse = th.sparse.mm(qs, ks)
        return a_sparse

    # Compute softmax normalization
    @staticmethod
    def normalize_block_attention_matrix(a_sparse, inv_sqrt_dim : typing.Union[ th.FloatTensor, float ] = 1.) -> th.FloatTensor :
        a_sparse = inv_sqrt_dim * a_sparse   # scaled dot product
        a_sparse = th.sparse.softmax(a_sparse, dim=1) # Convert to CSR for predictable unfolding
        return a_sparse


    """
    The (sparse) multihead attention  block-matrix:
    |x,x,                |
    |x,x,                |
    |    x,x,            |
    |    x,x,            |
    |        x,x,x,      |
    |        x,x,x,      |
    |        x,x,x,      |
    |              x,x,x,|
    |              x,x,x,|
    |              x,x,x,|
    The sparse features block matrix :
    | 0, 1, 2, 6, 7, 8,12,13,14,                                                               |
    |18,19,20,24,25,26,30,31,32,                                                               |
    |                            3, 4, 5, 9,10,11,15,16,17,                                    |
    |                           21,22,23,27,28,29,33,34,35,                                    |
    |                                                      36,37,38,42,43,44,                  |
    |                                                      48,49,50,54,55,56,                  |
    |                                                      60,62,62,66,67,68,                  |
    |                                                                        39,40,41,45,46,47,|
    |                                                                        51,52,53,57,58,59,|
    |                                                                        63,64,65,69,70,71,|  
    The size: 
    n = [2, 3,]; s = [3, 2,]; v_dim = 6
    """
    # Flatten v matrix in multihead context
    @staticmethod
    def compute_multihead_features_indices_flatten(l : th.LongTensor, v_dim : int, nheads : int) -> th.LongTensor :
        assert not v_dim % nheads, "The dimension of the features tensor should be divisible by the number of heads."
        hv_dim = v_dim // nheads
        offset_vf = th.cat((th.zeros(1, dtype=th.long, device=l.device), v_dim * th.cumsum(l, dim=0)[:-1]), dim=0).repeat_interleave(v_dim * l, dim=0)
        indx_vf = th.arange(v_dim * l.sum(dtype=th.long), dtype=l.dtype, device=l.device)
        offset_c_vf = ((indx_vf - offset_vf) // v_dim) * hv_dim + indx_vf % hv_dim
        block_id_vf = ((indx_vf % v_dim) // hv_dim)
        block_sz_vf = (l * hv_dim).repeat_interleave(l * v_dim)
        indx_vf = offset_vf + offset_c_vf + block_id_vf * block_sz_vf
        return indx_vf
    @staticmethod
    def construct_multihead_features_flatten(indx_vf : th.LongTensor, v : th.FloatTensor) -> th.FloatTensor :
        flatten_v = th.empty(v.nelement(), dtype=v.dtype, device=v.device)
        flatten_v[indx_vf] = v.view(-1)[:]
        return flatten_v

    # Unpack block-diagonal sparse matrix
    @staticmethod
    def compute_block_features_indices_flatten(n : th.LongTensor, s : th.LongTensor, v_dim : int, nheads : int) -> th.LongTensor :
        assert not v_dim % nheads, "The dimension of the features tensor should be divisible by the number of heads."
        hv_dim = v_dim // nheads
        # Construct permutation index
        offset_vb   = th.cat((th.zeros(1, dtype=th.long, device=n.device), th.cumsum(n * s * v_dim, dim=0)[:-1]), dim=0).repeat_interleave(v_dim * n * s, dim=0)
        indx_vb     = th.arange(v_dim * (n * s).sum(dtype=th.long), dtype=n.dtype, device=n.device)
        block_id_vb = ((indx_vb - offset_vb) // (s * hv_dim).repeat_interleave(n * s * v_dim, dim=0))
        indx_col_vb = ((indx_vb - offset_vb) %  (s * hv_dim).repeat_interleave(n * s * v_dim, dim=0))
        flatten_sz  = (s * v_dim).repeat_interleave(n * s * v_dim)
        inner_offset  = (block_id_vb %  (n.repeat_interleave(n * s * v_dim, dim=0))) * flatten_sz
        stride_offset = (block_id_vb // (n.repeat_interleave(n * s * v_dim, dim=0))) * flatten_sz // nheads
        indx_vfs = offset_vb + inner_offset + stride_offset + indx_col_vb
        return indx_vfs
    @staticmethod
    def construct_block_features_flatten(indx_vfs : th.LongTensor, v_sparse : th.FloatTensor) -> th.FloatTensor :
        flatten_v = th.empty(indx_vfs.nelement(), dtype=v_sparse.dtype, device=v_sparse.device)
        flatten_v[indx_vfs] = v_sparse.values()[:]
        return flatten_v

    # Compute sparse features product
    @staticmethod
    def compute_block_features_indices_product(n : th.LongTensor, l : th.LongTensor, s : th.LongTensor, v_dim : int, nheads : int) -> th.LongTensor :
        assert not v_dim % nheads , "The dimension of the features tensor should be divisible by the number of heads."
        hv_dim = v_dim // nheads
        tz = th.zeros(1, dtype=th.long, device=n.device)
        # Construct sparse matrix
        indx_vs     = th.arange((n * s).sum(dtype=th.long) * v_dim, dtype=th.long, device=n.device)
        offset_vs   = th.cat((tz, th.cumsum((s * v_dim).repeat_interleave(n, dim=0), dim=0)[:-1]), dim=0).repeat_interleave(l * v_dim, dim=0)
        indx_col_vs = (indx_vs - offset_vs) %  (s * hv_dim).repeat_interleave(n * s * v_dim, dim=0)
        block_id_vs = (indx_vs - offset_vs) // (s * hv_dim).repeat_interleave(n * s * v_dim, dim=0)
        offset_j_vs = th.cat((tz, th.cumsum(s * v_dim, dim=0)[:-1]), dim=0).repeat_interleave(n * s * v_dim)
        indx_j_vs   = offset_j_vs + block_id_vs * (s * hv_dim).repeat_interleave(n * s * v_dim, dim=0) + indx_col_vs
        offset_i_vs = nheads * th.cat((tz, th.cumsum(n, dtype=th.long, dim=0)[:-1]), dim=0).repeat_interleave(n * s * v_dim, dim=0)
        indx_row_vs = (th.arange(n.sum(dtype=th.long), dtype=th.long, device=n.device) - th.cat((tz, th.cumsum(n, dim=0)[:-1]), dim=0).repeat_interleave(n, dim=0)).repeat_interleave((s * v_dim).repeat_interleave(n, dim=0), dim=0)
        indx_i_vs   = offset_i_vs + block_id_vs * n.repeat_interleave(n * s * v_dim) + indx_row_vs
        # Return the indices
        indx_vs = th.stack((indx_i_vs, indx_j_vs), dim=0)
        return indx_vs
    @staticmethod
    def construct_block_features_product(n : th.LongTensor, s : th.LongTensor, v_dim : int, nheads :int, indx_vs : th.LongTensor, flatten_v : th.FloatTensor, a_sparse : th.FloatTensor) -> th.FloatTensor :
        # Compute the product with attention matrix
        v_sparse = th.sparse_coo_tensor(indx_vs, flatten_v, (nheads * n.sum(dtype=th.long), s.sum(dtype=th.long) * v_dim))
        return v_sparse.coalesce()

    # Attention matrix is computed as sparse block-diagonal operation
    @staticmethod
    def block_attention ( u_dim : int,
                          v_dim : int,
                          nheads : int,
                          n : th.LongTensor,
                          s : th.LongTensor,
                          q : th.FloatTensor,
                          k : th.FloatTensor,
                          v : th.FloatTensor,
                          indx_sa  : th.LongTensor, # indx_sa = BlockTransformer.compute_block_attention_indices(n, u_dim, nheads)
                          indx_vf  : th.LongTensor, # indx_vf = BlockTransformer.compute_multihead_features_indices_flatten(l, v_dim, nheads)
                          indx_vs  : th.LongTensor, # indx_vs = BlockTransformer.compute_block_features_indices_product(n, l, s, v_dim, nheads)
                          indx_vsf : th.LongTensor, # indx_vsf = BlockTransformer.compute_block_features_indices_flatten(n, s, v_dim, nheads)
                          inv_sqrt_dim : float = 1.,
                        ) -> th.FloatTensor :
        # Compute attention matrix
        a_sparse = BlockTransformer.compute_block_attention_matrix(n, q, k, indx_sa, u_dim, nheads=nheads)

        # Compute softmax normalization
        a_sparse_n = BlockTransformer.normalize_block_attention_matrix(a_sparse, inv_sqrt_dim=inv_sqrt_dim)

        # Flatten multihead tensor
        flatten_vs = BlockTransformer.construct_multihead_features_flatten(indx_vf, v)
        # Block-diagonal multihead tensor
        v_sparse = BlockTransformer.construct_block_features_product(n, s, v_dim, nheads, indx_vs, flatten_vs, a_sparse)

        # Compute product of attention matrix and features matrix
        vo_sparse = th.sparse.mm(a_sparse_n, v_sparse)
        #th.cuda.empty_cache() # Manually release unused memory

        # Unpack sparse output matrix vo
        # Reverse copy from block-matrix into flatten array
        flatten_vos = BlockTransformer.construct_block_features_flatten(indx_vsf, vo_sparse)
        # Unflatten multihead tensor
        vos = flatten_vos[indx_vf].view(-1, v_dim)

        return vos

    # Attention matrix is computed as fast dense operation.
    # It is applicable only for 1-smaple cases
    @staticmethod
    def dense_attention ( u_dim : int,
                          v_dim : int,
                          nheads : int,
                          n : th.LongTensor,
                          s : th.LongTensor,
                          q_dense : th.FloatTensor,
                          k_dense : th.FloatTensor,
                          v_dense : th.FloatTensor,
                          inv_sqrt_dim : float = 1.,
                        ) -> th.FloatTensor :
        # Compute attention matrix
        a_dense = th.matmul(q_dense.view(q_dense.shape[0], nheads, -1).permute(1, 0, 2),
                            k_dense.view(k_dense.shape[0], nheads, -1).permute(1, 2, 0))

        # Compute softmax normalization
        a_dense_n = th.nn.functional.softmax(a_dense / inv_sqrt_dim, dim=2)

        # Compute product of attention matrix and features matrix
        vo_dense = th.matmul(a_dense_n, v_dense.view(n[0], v_dense.shape[0] // n[0], nheads, -1).permute(2, 0, 1, 3).reshape(nheads, n[0], -1))

        # Output in a proper dimensions
        return vo_dense.view(nheads, n[0], v_dense.shape[0] // n[0], -1).permute(1, 2, 0, 3).reshape(v_dense.shape[0], -1)



# The unittest for sparse matrix-matrix multiplication
def unittest_block_attention(u_dim : int, v_dim : int, nheads : int, n : th.LongTensor, l : th.LongTensor, v : th.FloatTensor) -> bool :
    # Create sparse and dense matrices
    assert not u_dim % nheads, "u_dim must be divisible by nheads"
    hu_dim = u_dim // nheads
    assert not v_dim % nheads, "v_dim must be divisible by nheads"
    hv_dim = v_dim // nheads
    assert n.device == l.device and n.device == v.device , "All tensors (n, l and v) should be at the same device."
    inv_sqrt_dim = th.FloatTensor([1. / float(u_dim),], device=n.device).sqrt().item()
    # Dense query block
    q = th.arange(n.sum(dtype=th.long) * u_dim, dtype=th.float32, device=n.device).view(-1, u_dim)
    q_dense = th.zeros((n.sum(dtype=th.long) * nheads, n.nelement() * u_dim), dtype=th.float32, device=n.device)
    offset_zy = 0 ; offset_zx = 0 ; offset_u = 0
    for ni in n.numpy() :
        for hi in range(nheads) :
            q_dense[offset_zy : offset_zy + ni, offset_zx + hi * hu_dim : offset_zx + hi * hu_dim + hu_dim] = q[offset_u : offset_u + ni, hi * hu_dim : hi * hu_dim + hu_dim]
            offset_zy = offset_zy + ni
        offset_zx = offset_zx + u_dim
        offset_u = offset_u + ni
    # Dense keys block
    k_dense = th.zeros((n.sum(dtype=th.long) * nheads, n.nelement() * u_dim), dtype=th.float32, device=n.device)
    k = th.flip(q, [0, 1])
    offset_zx = 0 ; offset_zy = 0 ; offset_u = 0
    for ni in n.numpy() :
        for hi in range(nheads) :
            k_dense[offset_zy : offset_zy + ni, offset_zx + hi * hu_dim : offset_zx + hi * hu_dim + hu_dim] = k[offset_u : offset_u + ni, hi * hu_dim : hi * hu_dim + hu_dim]
            offset_zy = offset_zy + ni
        offset_zx = offset_zx + u_dim
        offset_u = offset_u + ni
    # Compute attention matrix
    kt_dense = th.transpose(k_dense, 0, 1)
    a_dense = th.matmul(q_dense, kt_dense)
    indx_sa = BlockTransformer.compute_block_attention_indices(n, u_dim, nheads)
    a_sparse = BlockTransformer.compute_block_attention_matrix(n, q, k, indx_sa, u_dim, nheads=nheads)

    if not th.all(th.isclose(a_dense, a_sparse.to_dense(), rtol=1e-06, atol=1e-08)) :
        print("Discrepancy in the unnormalized attention matrix")
        return False

    # Compute softmax normalization
    a_dense_n = a_dense * inv_sqrt_dim  # scaled dot product
    offset_n = 0
    for ni in n.numpy() :
        for hi in range(nheads) :
            a_dense_n[offset_n : offset_n + ni, : offset_n] = float('-inf')
            a_dense_n[offset_n : offset_n + ni, offset_n + ni :] = float('-inf')
            offset_n = offset_n + ni
    a_dense_n = th.nn.functional.softmax(a_dense_n)

    a_sparse_n = BlockTransformer.normalize_block_attention_matrix(a_sparse, inv_sqrt_dim=inv_sqrt_dim)

    if not th.all(th.isclose(a_dense_n, a_sparse_n.to_dense(), rtol=1e-02, atol=1e-06)) :
        print("Discrepancy in the normalized attention matrix")
        return False

    # Dense values block
    s = l[th.cat((th.zeros(1, dtype=th.long, device=n.device), th.cumsum(n, dtype=th.long, dim=0)[:-1]), dim=0)]
    # Flatten multihead tensor
    flatten_vd = th.FloatTensor(v.nelement(), device=n.device)
    offset_vf = 0 ; offset_vy = 0
    for (i, si) in enumerate(s.numpy()) :
        for _ in range(n[i].item()) :
            for hi in range(nheads) :
                flatten_vd[offset_vf : offset_vf + si * hv_dim] = v[offset_vy : offset_vy + si, hi * hv_dim : hi * hv_dim + hv_dim].flatten()
                offset_vf = offset_vf + si * hv_dim
            offset_vy = offset_vy + si

    indx_vf = BlockTransformer.compute_multihead_features_indices_flatten(l, v_dim, nheads)
    flatten_vs = BlockTransformer.construct_multihead_features_flatten(indx_vf, v)

    if th.any(flatten_vd != flatten_vs) :
        print("Discrepancy in the flatten features")
        return False

    # Construct v_dense
    # Block-diagonal multihead tensor
    v_dense = th.zeros(nheads * n.sum(dtype=th.long), s.sum(dtype=th.long) * v_dim, dtype=th.float32, device=n.device)
    offset_vf = 0 ; offset_vy = 0 ; offset_vx = 0
    for (ni, si) in zip(n.numpy(), s.numpy()) :
        v_sample = flatten_vd[offset_vf : offset_vf + ni * si * v_dim].view(ni, nheads, -1).transpose(0, 1).reshape(nheads, ni, si * hv_dim)
        for hi in range(nheads) :
            v_dense[offset_vy : offset_vy + ni, offset_vx : offset_vx + si * hv_dim] = v_sample[hi, ...]
            offset_vy = offset_vy + ni
            offset_vx = offset_vx + si * hv_dim
        offset_vf = offset_vf + ni * si * v_dim
        del v_sample

    indx_vs = BlockTransformer.compute_block_features_indices_product(n, l, s, v_dim, nheads)
    v_sparse = BlockTransformer.construct_block_features_product(n, s, v_dim, nheads, indx_vs, flatten_vs, a_sparse)

    if th.any(v_dense != v_sparse.to_dense()) :
        print("Discrepancy in the block-diagonal features")
        return False

    # Compute product of attention matrix and features matrix
    vo_dense = th.matmul(a_dense_n, v_dense)
    vo_sparse = th.sparse.mm(a_sparse_n, v_sparse)

    if not th.all(th.isclose(vo_sparse.to_dense(), vo_dense, rtol=1e-06, atol=1e-08)) :
        print("Discrepancy in updated features")
        return False

    # Unpack sparse matrix vo
    # Reverse copy from block-matrix into flatten array
    flatten_vod = th.FloatTensor(flatten_vd.nelement(), device=v.device)
    offset_vf = 0 ; offset_vy = 0 ; offset_vx = 0
    for (ni, si) in zip(n.numpy(), s.numpy()) :
        iv_sample = th.FloatTensor(ni * si * v_dim, device=n.device).view(nheads, ni, si * hv_dim)
        for hi in range(nheads) :
            iv_sample[hi, ...] = vo_dense[offset_vy : offset_vy + ni, offset_vx : offset_vx + si * hv_dim]
            offset_vy = offset_vy + ni
            offset_vx = offset_vx + si * hv_dim
        flatten_vod[offset_vf : offset_vf + ni * si * v_dim] = iv_sample.transpose(0, 1).flatten()[:]
        offset_vf = offset_vf + ni * si * v_dim
        del iv_sample

    indx_vofs = BlockTransformer.compute_block_features_indices_flatten(n, s, v_dim, nheads)
    flatten_vos = BlockTransformer.construct_block_features_flatten(indx_vofs, vo_sparse)

    if not th.all(th.isclose(flatten_vos, flatten_vod, rtol=1e-06, atol=1e-08)) :
        print("Discrepancy in the unblocking flattened features")
        return False

    # Unflatten multihead tensor
    vod = th.FloatTensor(v.nelement(), device=n.device).view(-1, v_dim)
    offset_vf = 0 ; offset_vy = 0
    for (i, si) in enumerate(s.numpy()) :
        for _ in range(n[i].item()) :
            for hi in range(nheads) :
                vod[offset_vy : offset_vy + si, hi * hv_dim : hi * hv_dim + hv_dim] = flatten_vod[offset_vf : offset_vf + si * hv_dim].view(-1, hv_dim)[...]
                offset_vf = offset_vf + si * hv_dim
            offset_vy = offset_vy + si

    vos = flatten_vos[indx_vf].view(-1, v_dim)

    if not th.all(th.isclose(vos, vod, rtol=1e-06, atol=1e-08)) :
        print("Discrepancy in the unflattened features")
        return False

    return True



if __name__ == "__main__" :

    # Run block transformer on collection of small problems
    import random # For random problem generation
    device = th.device("cpu")

    NRUNS = 128
    # The toy debbuging case
    nheads = 2
    u_dim = 2 * nheads
    v_dim = 3 * nheads
    n = th.LongTensor([2, 3, ], device=device)
    s = th.LongTensor([3, 2, ], device=device)
    # DEBUGING STUFF
    #n_heads = 2 ; u_dim = 4 ; v_dim = 2 ; n = th.LongTensor([2, 2, 4,], device=device) ; s = th.LongTensor([3, 4, 2,], device=device)
    run = 0
    while run != NRUNS :
        if run :
            # Generate new random problem
            nheads = random.randint(1, 5)
            u_dim = random.randint(1, 6) * nheads
            v_dim = random.randint(1, 6) * nheads
            size = random.randint(1, 8)
            n_list = [None] * size ; s_list = [None] * size
            for i in range(size):
                n_list[i] = random.randint(1, 6)
                s_list[i] = random.randint(1, 9)
            n = th.LongTensor(n_list, device=device)
            s = th.LongTensor(s_list, device=device)

        l = s.repeat_interleave(n, dim=0)
        v = th.arange(l.sum(dtype=th.long) * v_dim, dtype=th.float32, device=device).view(-1, v_dim)
        result = True # unittest_block_attention(u_dim, v_dim, nheads, n, l, v)
        if result :
            print("Unittest unittest_block_attention PASSED u_dim={:d} v_dim={:d} nheads={:d} n={!r} s={!r}".format(u_dim, v_dim, nheads, n, s))
            run = run + 1
        else      :
            print("Unittest unittest_block_attention FAILED u_dim={:d} v_dim={:d} nheads={:d} n={!r} s={!r}".format(u_dim, v_dim, nheads, n, s))
            exit(1)
    print("{:s}uccessfully finished {:d} testing runs with random small problems".format( "S" if run == NRUNS else "Uns", NRUNS))

    # Run forward step for block transformer and one sample with dense transform
    z_dim = 8
    nheads = 2
    n_list = [3]
    s_list = [2]
    z_dim_ = z_dim + (0 if not z_dim % nheads else nheads - z_dim % nheads)
    # Features
    n = th.LongTensor(n_list, device=device)
    s = th.LongTensor(s_list, device=device)
    l = s.repeat_interleave(n, dim=0)
    # Create attention matrices
    q = th.arange(n.sum(dtype=th.long) * z_dim_, dtype=th.float32, device=device).view(-1, z_dim_)
    k = th.arange(n.sum(dtype=th.long) * z_dim_, dtype=th.float32, device=device).view(-1, z_dim_)
    v = th.arange(l.sum(dtype=th.long) * z_dim_, dtype=th.float32, device=device).view(-1, z_dim_)
    run = 0
    while run != NRUNS :
        if run :
            # Pick a random system of ONE sample
            z_dim = random.randint(32, 256)
            nheads = random.randint(2, 8)
            n_list = [None]
            s_list = [None]
            n_list[0] = random.randint(1, 12)
            s_list[0] = random.randint(1, 36)
            z_dim_ = z_dim + (0 if not z_dim % nheads else nheads - z_dim % nheads)
            # Features
            n = th.LongTensor(n_list, device=device)
            s = th.LongTensor(s_list, device=device)
            l = s.repeat_interleave(n, dim=0)
            # Create attention matrices
            q = th.rand(n.sum(dtype=th.long) * z_dim_, dtype=th.float32, device=device).view(-1, z_dim_)
            k = th.rand(n.sum(dtype=th.long) * z_dim_, dtype=th.float32, device=device).view(-1, z_dim_)
            v = th.rand(l.sum(dtype=th.long) * z_dim_, dtype=th.float32, device=device).view(-1, z_dim_)
        # BlockTransformer indices
        s = l[th.cat((th.zeros(1, dtype=th.long, device=n.device), th.cumsum(n, dtype=th.long, dim=0)[:-1]), dim=0)]
        sc_indx = th.arange(l.nelement(), dtype=th.long, device=l.device).repeat_interleave(l, 0)
        indx_sa = BlockTransformer.compute_block_attention_indices(n, z_dim_, nheads)
        indx_vf = BlockTransformer.compute_multihead_features_indices_flatten(l, z_dim_, nheads)
        indx_vs = BlockTransformer.compute_block_features_indices_product(n, l, s, z_dim_, nheads)
        indx_vsf = BlockTransformer.compute_block_features_indices_flatten(n, s, z_dim_, nheads)
        # Block transformer attention
        result_block = BlockTransformer.block_attention ( z_dim_, z_dim_, nheads,
                                                          n, s, q, k, v,
                                                          indx_sa, indx_vf, indx_vs, indx_vsf,
                                                          inv_sqrt_dim = th.FloatTensor([1.,], device=device).sqrt().item() )
        result_dense = BlockTransformer.dense_attention ( z_dim_, z_dim_, nheads,
                                                          n, s, q, k, v,
                                                          inv_sqrt_dim = th.FloatTensor([1.,], device=device).sqrt().item() )
        result = th.allclose(result_block, result_dense, atol=1e-06)
        if result :
            print("Unittest single dense_block_attention PASSED u_dim={:d} v_dim={:d} nheads={:d} n={!r} s={!r}".format(u_dim, v_dim, nheads, n, s))
            run = run + 1
        else      :
            print("Unittest single dense_block_attention FAILED u_dim={:d} v_dim={:d} nheads={:d} n={!r} s={!r}".format(u_dim, v_dim, nheads, n, s))
            exit(1)
    print("{:s}uccessfully finished {:d} testing runs with random small problems".format( "S" if run == NRUNS else "Uns", NRUNS))

    # Run training short session for multiple-samples block-diagonal attention
    # Create model
    u_dim = 64
    v_dim = 80
    z_dim = 256
    nheads = 16
    model = BlockTransformer(u_dim, v_dim, z_dim, nheads)
    # Creeate tensors
    n = th.LongTensor([ 2,  3,  5,  4], device=th.device("cpu"))
    l = th.LongTensor([28, 13, 45, 54], device=th.device("cpu")).repeat_interleave(n, dim=0)
    u = th.rand(n.sum(dtype=th.long), u_dim, dtype=th.float32, device=th.device("cpu"), requires_grad=True)
    v = th.rand(l.sum(dtype=th.long), v_dim, dtype=th.float32, device=th.device("cpu"), requires_grad=True)
    # BlockTransformer indices
    s = l[th.cat((th.zeros(1, dtype=th.long, device=n.device), th.cumsum(n, dtype=th.long, dim=0)[:-1]), dim=0)]
    sc_indx = th.arange(l.nelement(), dtype=th.long, device=l.device).repeat_interleave(l, 0)
    indx_sa = BlockTransformer.compute_block_attention_indices(n, z_dim, nheads)
    indx_vf = BlockTransformer.compute_multihead_features_indices_flatten(l, z_dim, nheads)
    indx_vs = BlockTransformer.compute_block_features_indices_product(n, l, s, z_dim, nheads)
    indx_vsf = BlockTransformer.compute_block_features_indices_flatten(n, s, z_dim, nheads)
    # Evaluate model
    vo = model.forward(n, s, l, u, v,        sc_indx, indx_sa, indx_vf, indx_vs, indx_vsf)
    loss = vo.norm()
    # Backward propagate
    loss.backward()
    print("Successfully finished block-diagonal attention back-propagation of multiple samples with Loss = {:f}".format(loss.item()))
    # Run training short session for single-sample dense attention
    # Creeate tensors
    n = th.LongTensor([5], device=th.device("cpu"))
    l = th.LongTensor([27], device=th.device("cpu")).repeat_interleave(n, dim=0)
    u = th.rand(n.sum(dtype=th.long), u_dim, dtype=th.float32, device=th.device("cpu"), requires_grad=True)
    v = th.rand(l.sum(dtype=th.long), v_dim, dtype=th.float32, device=th.device("cpu"), requires_grad=True)
    # BlockTransformer indices
    sc_indx = th.arange(l.nelement(), dtype=th.long, device=l.device).repeat_interleave(l, 0)
    vo = model.forward(n, s, l, u, v,        sc_indx, None, None, None, None)
    loss = vo.norm()
    # Backward propagate
    loss.backward()
    print("Successfully finished dense attention back-propagation of single sample with Loss = {:f}".format(loss.item()))
