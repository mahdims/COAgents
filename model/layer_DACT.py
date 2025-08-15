from torch import nn
import torch
import torch.nn.functional as F
import math
import torch as th
from torch.nn.modules.module import Module
from datastructures import Sample, VRP_MAX_NODES, VRP_MAX_DUMMY_DEPOTS
# Adapted from https://github.com/yining043/VRP-DACT/blob/new_version/nets/graph_layers.py

class MultiHeadDACAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        vn_dim,    # Dimension of node features (NFE)
        vp_dim,    # Dimension of positional features (PFE)
        val_dim=None,
        key_dim=None
    ):
        super(MultiHeadDACAttention, self).__init__()
        
        self.n_heads = n_heads
        self.vn_dim = vn_dim
        self.vp_dim = vp_dim
    
        
        # Calculate key/value dimensions per head
        self.key_dim = key_dim if key_dim is not None else (vn_dim + vp_dim) // n_heads
        self.val_dim = self.key_dim if val_dim is None else val_dim
        
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        # Node feature transformations
        self.W_query_node = nn.Parameter(torch.Tensor(n_heads, 2 * self.vn_dim, self.key_dim))
        self.W_key_node = nn.Parameter(torch.Tensor(n_heads, 2 * self.vn_dim, self.key_dim))
        self.W_val_node = nn.Parameter(torch.Tensor(2 * n_heads, 2 * self.vn_dim, self.val_dim))
        
        # Positional feature transformations
        self.W_query_pos = nn.Parameter(torch.Tensor(n_heads, 2 * self.vp_dim, self.key_dim))
        self.W_key_pos = nn.Parameter(torch.Tensor(n_heads, 2 * self.vp_dim, self.key_dim))
        self.W_val_pos = nn.Parameter(torch.Tensor(2 * n_heads, 2 * self.vp_dim, self.val_dim))
        
        # Output projections
        self.W_out_node = nn.Parameter(torch.Tensor(n_heads, 2 * self.val_dim, 2 * self.vn_dim))
        self.W_out_pos = nn.Parameter(torch.Tensor(n_heads, 2 * self.val_dim, 2 * self.vp_dim))
        
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_node, h_pos):

        batch_size, graph_size, _ = h_node.size()
        
        # Reshape 
        h_node = h_node.contiguous().view(-1, 2 * self.vn_dim)  # (batch*graph_size, 2*vn_dim)
        h_pos = h_pos.contiguous().view(-1, 2 * self.vp_dim)    # (batch*graph_size, 2*vp_dim)

        # Projections
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_v = (2, self.n_heads, batch_size, graph_size, -1)
        
        # Node computations
        Q_node = torch.matmul(h_node, self.W_query_node).view(shp)
        K_node = torch.matmul(h_node, self.W_key_node).view(shp)
        V_node = torch.matmul(h_node, self.W_val_node).view(shp_v)
        
        # Position computations
        Q_pos = torch.matmul(h_pos, self.W_query_pos).view(shp)
        K_pos = torch.matmul(h_pos, self.W_key_pos).view(shp)
        V_pos = torch.matmul(h_pos, self.W_val_pos).view(shp_v)

        # Attention computations
        node_correlations = self.norm_factor * torch.matmul(Q_node, K_node.transpose(2, 3))
        pos_correlations = self.norm_factor * torch.matmul(Q_pos, K_pos.transpose(2, 3))
        
        attn1 = F.softmax(node_correlations, dim=-1)
        attn2 = F.softmax(pos_correlations, dim=-1)

        # Cross-aspect mixing
        heads_node_1 = torch.matmul(attn1, V_node[0])  # Self-attention
        heads_node_2 = torch.matmul(attn2, V_node[1])  # cross-aspect
        heads_node = torch.cat([heads_node_1, heads_node_2], -1)
        
        heads_pos_1 = torch.matmul(attn1, V_pos[0])    # Node-guided
        heads_pos_2 = torch.matmul(attn2, V_pos[1])    # cross-aspect
        heads_pos = torch.cat([heads_pos_1, heads_pos_2], -1)

        # Output projections
        out_node = self.projection(heads_node, self.W_out_node, batch_size, graph_size, 2 * self.vn_dim)
        out_pos = self.projection(heads_pos, self.W_out_pos, batch_size, graph_size, 2 * self.vp_dim)
        return out_node, out_pos

    def projection(self, heads, W_out, batch_size, graph_size, embed_dim):
        return torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * 2 * self.val_dim),
            W_out.view(-1, embed_dim)
        ).view(batch_size, graph_size, embed_dim)
        

class AttentionSubLayer(nn.Module):
    def __init__(self, n_heads, vn_dim, vp_dim):
        super().__init__()
        self.attn = MultiHeadDACAttention(n_heads, vn_dim, vp_dim)
        self.norm_node = nn.LayerNorm(2 * vn_dim)
        self.norm_pos = nn.LayerNorm(2 * vp_dim)

    def forward(self, node_feats, pos_feats):
        attn_node, attn_pos = self.attn(node_feats, pos_feats)
        return (
            self.norm_node(node_feats + attn_node),
            self.norm_pos(pos_feats + attn_pos)
        )


# FFN sublayer
class FFNSubLayer(nn.Module):
    def __init__(self, vn_dim, vp_dim, ff_hidden):
        super().__init__()
        self.ff_node = nn.Sequential(
            nn.Linear(2 * vn_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, vn_dim)
        )
        self.ff_pos = nn.Sequential(
            nn.Linear(2 * vp_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, vp_dim)
        )
        self.norm_node = nn.LayerNorm(vn_dim)
        self.norm_pos = nn.LayerNorm(vp_dim)

    def forward(self, node_feats, pos_feats):
        return (
            self.norm_node(self.ff_node(node_feats)),
            self.norm_pos(self.ff_pos(pos_feats))
        )
    

# Full x layer
class DACTEncoder(nn.Module):
    def __init__(
        self,
        n_heads,
        vn_dim,
        vp_dim,
        ff_hidden
    ):
        super().__init__()
        self.attn_layer = AttentionSubLayer(n_heads, vn_dim, vp_dim)
        self.ffn_layer = FFNSubLayer(vn_dim, vp_dim, ff_hidden)
        self.vn_dim = vn_dim
        self.vp_dim = vp_dim

    def forward(self, n, l, _v, v_):
        batch_size = sum(n).item()  
        n_nodes = l[0]      

        _v_nfe = _v[:, :self.vn_dim]
        _v_pfe = _v[:, self.vn_dim:]
        v_nfe = v_[:, :self.vn_dim]
        v_pfe = v_[:, self.vn_dim:]
        
        # Combine across versions
        h_node = torch.cat([_v_nfe, v_nfe], dim=1)  # [total, vn_dim*2]
        h_pos = torch.cat([_v_pfe, v_pfe], dim=1)   # [total, vp_dim*2]

        # Reshape to batch format
        node_feats = h_node.view(batch_size, n_nodes, 2 * self.vn_dim)  # [B, N, 2*vn_dim]
        pos_feats = h_pos.view(batch_size, n_nodes, 2 * self.vp_dim)    # [B, N, 2*vp_dim]
        # Attention phase
        attn_node, attn_pos = self.attn_layer(node_feats, pos_feats)
        # FFN phase
        out_node, out_pos = self.ffn_layer(attn_node, attn_pos)
        out_node = out_node.view(-1, self.vn_dim) 
        out_pos = out_pos.view(-1, self.vp_dim) 
        v = th.cat((out_node, out_pos), dim=1)
        return v
    








#================================================== DECODER ==============================================
# implements MLP module
class MLP(torch.nn.Module):
    def __init__(self,
                input_dim = 128,
                feed_forward_dim = 64,
                embedding_dim = 64,
                output_dim = 1
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.ReLU = nn.ReLU(inplace = True)
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result
    


# implements the multi-head compatibility layer
class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()
    
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask=None):
        if h is None:
            h = q 

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)


        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)  
        K = torch.matmul(hflat, self.W_key).view(shp)   

        # Calculate compatibility 
        compatibility = torch.matmul(Q, K.transpose(2, 3))
        
        return self.norm_factor * compatibility
    

# implements the DAC decoder
class DACTDecoder(nn.Module):
    def __init__(
            self,
            n_heads,
            vn_dim,
            vp_dim,
            val_dim=None,
            key_dim=None
    ):
        super(DACTDecoder, self).__init__()
        self.n_heads = n_heads
        self.vn_dim = vn_dim
        self.vp_dim = vp_dim
        self.C = 6

        
        # for MHC sublayer (NFE aspect)
        self.compater_node = MultiHeadCompat(n_heads,
                                        2*self.vn_dim,
                                        2*self.vn_dim,
                                        val_dim,
                                        key_dim)
        
        # for MHC sublayer (PFE aspect)
        self.compater_pos = MultiHeadCompat(n_heads,
                                2*self.vp_dim,
                                2*self.vp_dim,
                                val_dim,
                                key_dim)
        
        # for Max-Pooling sublayer
        self.project_graph_pos  = nn.Linear(2 * self.vp_dim, 2 * self.vp_dim, bias=False)
        self.project_graph_node = nn.Linear(2 * self.vn_dim, 2 * self.vn_dim, bias=False)
        self.project_node_pos   = nn.Linear(2 * self.vp_dim, 2 * self.vp_dim, bias=False)
        self.project_node_node  = nn.Linear(2 * self.vn_dim, 2 * self.vn_dim, bias=False)
    
        
        # for feed-forward aggregation (FFA)sublayer
        self.value_head = MLP(self.n_heads*2, 32, 32, 1)



    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
        
    def forward(self, n, l, v_, _v):
        #h_em, pos_em = v[:, :self.vn_dim], v[:, self.vn_dim:]
        batch_size = sum(n).item()
        graph_size = l[0]    
        _v_nfe = _v[:, :self.vn_dim]
        _v_pfe = _v[:, self.vn_dim:]
        v_nfe = v_[:, :self.vn_dim]
        v_pfe = v_[:, self.vn_dim:]
                # Combine across versions
        h_em = torch.cat([_v_nfe, v_nfe], dim=1)  # [total, vn_dim*2]
        pos_em = torch.cat([_v_pfe, v_pfe], dim=1)   # [total, vp_dim*2]
        # Reshape to batch format
        h_em = h_em.view(batch_size, graph_size, 2*self.vn_dim)  # [B, N, vn_dim]
        pos_em = pos_em.view(batch_size, graph_size, 2*self.vp_dim)    # [B, N, vp_dim]
        
        # Max-Pooling sublayer
        h_node_refined = self.project_node_node(h_em) + self.project_graph_node(h_em.max(1)[0])[:, None, :].expand(batch_size, graph_size, 2*self.vn_dim)
        h_pos_refined = self.project_node_pos(pos_em) + self.project_graph_pos(pos_em.max(1)[0])[:, None, :].expand(batch_size, graph_size, 2*self.vp_dim)
        
        # MHC sublayer
        compatibility                      = torch.zeros((batch_size, graph_size, graph_size, self.n_heads * 2), device = h_node_refined.device)
        compatibility[:,:,:,:self.n_heads] = self.compater_pos(h_pos_refined).permute(1,2,3,0)
        compatibility[:,:,:,self.n_heads:] = self.compater_node(h_node_refined).permute(1,2,3,0)
        # FFA sublater
        FFA_out = self.value_head(compatibility).squeeze(-1)
        Y = self.C*torch.tanh(FFA_out)
                
        # Create node mask (dummy depots except real depot)
        #mask = torch.zeros(graph_size, dtype=torch.bool, device=h_node_refined.device)
        #mask[:VRP_MAX_DUMMY_DEPOTS] = True
        #mask[VRP_MAX_DUMMY_DEPOTS-1] = False 
        #depots_mask = mask.unsqueeze(0) | mask.unsqueeze(1) 
        diagonal_mask = torch.eye(graph_size, dtype=torch.bool, device=h_node_refined.device)  # Diagonal mask
        #pair_mask = depots_mask | diagonal_mask  # Combine masks
        flat_mask = diagonal_mask.view(-1)

        #mask compatibility scores before softmax
        im = Y.view(batch_size, -1)  # [batch_size, graph_size*graph_size]
        #im_masked = im.masked_fill(flat_mask, -float('inf')) 
#
        #Y_dist = F.softmax(im_masked, dim=-1)

        return im #Y_dist
    


             




        
#def logsumexp(inputs, dim=None, keepdim=False):
#    """Numerically stable logsumexp.
#
#    Args:
#        inputs: A Variable with any shape.
#        dim: An integer.
#        keepdim: A boolean.
#
#    Returns:
#        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
#    """
#    # For a 1-D array x (any array along a single dimension),
#    # log sum exp(x) = s + log sum exp(x - s)
#    # with s = max(x) being a common choice.
#    if dim is None:
#        inputs = inputs.view(-1)
#        dim = 0
#    s, _ = torch.max(inputs, dim=dim, keepdim=True)
#    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
#    if not keepdim:
#        outputs = outputs.squeeze(dim)
#    return outputs
#
#class Sinkhorn(Module):
#    """
#    SinkhornNorm layer from https://openreview.net/forum?id=Byt3oJ-0W
#    
#    If L is too large or tau is too small, gradients will disappear 
#    and cause the network to NaN out!
#    """    
#    def __init__(self, sinkhorn_iters=5, tau=0.01):
#        super(Sinkhorn, self).__init__()
#        self.tau = tau
#        self.sinkhorn_iters = sinkhorn_iters
#
#    def row_norm(self, x):
#        """Unstable implementation"""
#        #y = torch.matmul(torch.matmul(x, self.ones), torch.t(self.ones))
#        #return torch.div(x, y)
#        """Stable, log-scale implementation"""
#        return x - logsumexp(x, dim=2, keepdim=True)
#
#    def col_norm(self, x):
#        """Unstable implementation"""
#        #y = torch.matmul(torch.matmul(self.ones, torch.t(self.ones)), x)
#        #return torch.div(x, y)
#        """Stable, log-scale implementation"""
#        return x - logsumexp(x, dim=1, keepdim=True)
#
#    def forward(self, x, eps=1e-6):
#        """ 
#            x: [batch_size, N, N]
#        """
#        x = x / self.tau
#        for _ in range(self.sinkhorn_iters):
#            x = self.row_norm(x)
#            x = self.col_norm(x)
#        return torch.exp(x) + eps
        #Y_dist = Y_dist.view(batch_size, graph_size, -1)
        #sinkhorn_part = self.sinkhorn.forward(Y_dist[:, VRP_MAX_DUMMY_DEPOTS:, VRP_MAX_DUMMY_DEPOTS:])
        #Y_dist = torch.cat([
        #Y_dist[:, :VRP_MAX_DUMMY_DEPOTS, :], 
        #torch.cat([
        #    Y_dist[:, VRP_MAX_DUMMY_DEPOTS:, :VRP_MAX_DUMMY_DEPOTS],  
        #    sinkhorn_part  
        #], dim=2)  
        #], dim=1)  
        #
        #Y_dist = Y_dist.view(batch_size, -1)