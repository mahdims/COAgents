import typing
import numpy as np
import torch as th
from layerGatedGCN import GatedGCNLayer
from layerCyclicFeatureEmbedding import CyclicEmbeddingLayer
from layerGraphEmbedding import GraphEmbeddingLayer
from layerMaskedBlockTransformer import MaskedBlockTransformer
from layerLossBCE import LossBCELayer
from datastructures import Sample, VRP_MAX_NODES, VRP_MAX_DUMMY_DEPOTS



# Size of the batch
BATCH_SIZE = 48
ACCUMULATE_GRAD_STEPS = 1
# Learning rate
PARAM_LEARNING_RATE=0.0001

# Create model
PARAM_U_DIM   = 64
PARAM_V_DIM   = 96
PARAM_EZ_DIM  = 32
PARAM_GGCN0_Z_DIM = 128
PARAM_GGCN0_DROP  = 0.1
PARAM_LOSS_Z_DIM  = 128

PARAM_TRANS0_NHEAD = 16
PARAM_TRANS0_Z_DIM = 256
PARAM_TRANS0_DROP  = 0.1



PARAM_DACT_NHEAD = 4
PARAM_DACT_HIDDEN = 128

# Heuristics model
class HeuristicNet(th.nn.Module) :
    def __init__ (self, ui_dim : int,   # Input features for global embeddings
                        up_dim : int,   # Input positions for global embeddings
                        u_dim  : int,   # Embedding dimension of the global embeddings
                        vi_dim : int,   # Input features for local embeddings
                        vp_dim : int,   # Input positions for local embeddings
                        v_dim  : int,   # Embedding dimension of the local embeddings
                    nedge_type : int,   # Amount of known edge (move) types
                        ez_dim : int,   # Edges embedings size
                    ggcn_z_dim : int,   # Latent space of zero-th layer of GGCN model
                    ggcn_drop  : float, # Dropout of GGCN layers
             transformer_nhead : int,   # Number in heads in transformer
             transformer_z_dim : int,   # Inner dimension of transformer
           transformer_dropout : float, # Dropout of transformer layers
                    loss_z_dim : int,   # Size of loss layer hidden space
                loss_solutions : bool = True, # Flag for adding the loss over solutions
               loss_heuristics : bool = True, # Flag for adding the loss over heuristics
                 ) :
        super(HeuristicNet, self).__init__()
        self.device  = th.device('cpu')
        self.ui_dim  = ui_dim
        self.up_dim  = up_dim
        self.u_dim   = u_dim
        self.vi_dim  = vi_dim
        self.vp_dim  = vp_dim
        self.v_dim   = v_dim
        self.nedge_type = nedge_type
        self.ez_dim  = ez_dim
        self.ggcn_z_dim = ggcn_z_dim
        self.ggcn_drop  = ggcn_drop
        self.transformer_z_dim = transformer_z_dim
        self.transformer_nhead = transformer_nhead
        self.transformer_dropout = transformer_dropout
        self.loss_z_dim = loss_z_dim
        self.loss_solutions = loss_solutions
        self.loss_heuristics = loss_heuristics
        # The VRP inner structure solver/embedder
        self.cyclic_embedding = CyclicEmbeddingLayer(node_dim=vi_dim,
                                                     embedding_dim=v_dim - vp_dim,
                                                     num_nodes=VRP_MAX_NODES,
                                                     num_dummy_depots=VRP_MAX_DUMMY_DEPOTS,
                                                     option=1) # use the simplified FEs and dummpy depots have same positional embedding
        # The solution graph level embedder
        self.graph_embedding  = GraphEmbeddingLayer(ui_dim,     # input dimension
                                                    up_dim,     # position embedding dimension
                                                    u_dim,      # output dimension
                                                    nedge_type, # maximum of edges types
                                                    ez_dim)     # number of edge embeddings
        # The Gated Graph Convolution Net
        self.ggcnns = th.nn.ModuleList([
                                       GatedGCNLayer(u_dim, v_dim, ez_dim, ggcn_z_dim, dropout=ggcn_drop), # layer ggcn0
                                       GatedGCNLayer(u_dim, v_dim, ez_dim, ggcn_z_dim, dropout=ggcn_drop), # layer ggcn1
                                       GatedGCNLayer(u_dim, v_dim, ez_dim, ggcn_z_dim, dropout=ggcn_drop), # layer ggcn2
        ])
        # The Transformer Net
        self.transs = th.nn.ModuleList([
                                       MaskedBlockTransformer(u_dim, v_dim, transformer_z_dim, transformer_nhead, transformer_nhead, dropout=transformer_dropout),  # layer transformer0
                                       MaskedBlockTransformer(u_dim, v_dim, transformer_z_dim, transformer_nhead, transformer_nhead, dropout=transformer_dropout),  # layer transformer1
                                       MaskedBlockTransformer(u_dim, v_dim, transformer_z_dim, transformer_nhead, transformer_nhead, dropout=transformer_dropout),  # layer transformer2
        ])
        # The Loss layer
        self.loss_layer = LossBCELayer(nedge_type, ez_dim, u_dim, v_dim, loss_z_dim)
    # forward routine
    def forward(self, batch : Sample) -> typing.Union[ typing.Tuple[th.LongTensor, th.LongTensor], th.FloatTensor ] :
        # Precompute graph comvolution indices
        sc_indx  = th.arange(batch.n.sum(), dtype=th.long, device=batch.n.device).repeat_interleave(batch.l[0], 0)
        # Embed local features
        vf = self.cyclic_embedding.forward_fe(batch.v[:, : self.vi_dim])
        v  = th.cat((vf, batch.v[:, self.vi_dim :]), dim=1)
        # Embed gloable features
        (u, ez) = self.graph_embedding.forward(batch.u[:, : self.ui_dim], batch.u[:, self.ui_dim :], batch.e)
        n = batch.n ; l = batch.l ;  e = batch.e
        # Gated graph convolutions
        u_ = u ; v_ = v
        for (ggcnn, trans) in zip(self.ggcnns, self.transs) :
            # Gated Graph convolution
            (u_, v_) = ggcnn.forward(n, l[0].item(), u_, v_, ez, e, sc_indx)
            # Graph attention
            v_       = trans.forward(n, l[0].item(), u_, v_)
        # Compute labels and the loss
        # Precompute label_solutions (it is needed for label_heuristics)
        if batch.g_dst is not None:
            subgraph_indx = th.repeat_interleave(th.arange(n.nelement(), dtype=th.long, device=batch.g_dst.device), n)
            min_dist = th.scatter_reduce(th.zeros(n.nelement(), dtype=th.long, device=batch.g_dst.device, requires_grad=False),
                                        0, subgraph_indx, batch.g_dst, reduce='amin', include_self=False)
            label_solutions = (batch.g_dst == min_dist[subgraph_indx]).to(dtype=th.float32, device=batch.g_dst.device)
        # Labels for moves
        ee = self.graph_embedding.forward_edges(th.arange(self.nedge_type, dtype=th.long, device=ez.device))
        if self.loss_heuristics :
            label_heuristics = th.zeros(n.sum(), self.nedge_type + 1, dtype=th.float32, device=ez.device, requires_grad=False)
            label_heuristics[th.arange(n.sum(), dtype=th.long, device=ez.device, requires_grad=False), batch.m_bst] = 1
            label_heuristics = label_heuristics * label_solutions.unsqueeze(1)
        else                    :
            label_heuristics = None
        # Labels for solutions
        if not self.loss_solutions :
            label_solutions = None
        loss = self.loss_layer.forward(n, l, u_, v_, sc_indx, None if label_solutions  is None else label_solutions,
                                                              None if label_heuristics is None else label_heuristics[:, : self.nedge_type].flatten(), ee)
        return loss
