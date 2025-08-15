import typing

import numpy as np
import torch as th
from layerGatedGCN import GatedGCNLayer
from layerCyclicFeatureEmbedding import CyclicEmbeddingLayer
from layerGraphEmbedding import GraphEmbeddingLayer
#from layerBlockTransformer import BlockTransformer
from layerMaskedBlockTransformer import MaskedBlockTransformer
from layerLossBCE import LossBCELayer
from layer_DACT import DACTEncoder, DACTDecoder

from datastructures import Sample, VRP_MAX_NODES, VRP_MAX_DUMMY_DEPOTS


# Heuristics model
class E2ENet(th.nn.Module) :
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
                    dact_nhead : int,   # Number in heads in DACT
                   dact_hidden : int,   #Latent space of DACT FF
                    loss_z_dim : int,   # Size of loss layer hidden space
                loss_solutions : bool = True, # Flag for adding the loss over solutions
               loss_heuristics : bool = True, # Flag for adding the loss over heuristics
               mode            : int  = 0 # for trianing 
                 ) :
        super(E2ENet, self).__init__()
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
        self.mode = mode

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

               #X_layer 
        self.dacts = th.nn.ModuleList([
                                       DACTEncoder(dact_nhead, v_dim - vp_dim, vp_dim, dact_hidden),
                                       DACTEncoder(dact_nhead, v_dim - vp_dim, vp_dim, dact_hidden),
                                       DACTEncoder(dact_nhead, v_dim - vp_dim, vp_dim, dact_hidden),
        ])

        self.decoder = DACTDecoder(dact_nhead, v_dim - vp_dim, vp_dim)
        pos_weight = th.tensor(60)
        self.loss  = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight) #th.nn.BCELoss() 

        # Debugging
        #self.debug_layers = []

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
        u_ = u ; v_ = v; _v = v
        for (ggcnn, trans, dact) in zip(self.ggcnns, self.transs, self.dacts) :
            # Gated Graph convolution
            (u_, v_) = ggcnn.forward(n, l[0].item(), u_, v_, ez, e, sc_indx)
            # Graph attention
            v_       = trans.forward(n, l[0].item(), u_, v_)
            # DACT layer
            _v = dact.forward(n, l, v_, _v)
        comp = self.decoder.forward(n, l, v_, _v)
        #if self.mode == 0:
        #        B, num_candidates, H = batch.t.shape
        #        max_n = max(n)
        #        padded_predicted = th.zeros(B, max_n, H, device=batch.n.device, dtype=comp.dtype)
        #        valid_mask = th.zeros(B, max_n, device=batch.n.device, dtype=th.bool)
        #        
        #        start = 0
        #        for i, n_i in enumerate(n):
        #            padded_predicted[i, :n_i] = comp[start:start+n_i]
        #            valid_mask[i, :n_i] = True
        #            start += n_i
#
        #        # L1 Distance Computation
        #        distances = th.abs(padded_predicted.unsqueeze(2) - batch.t.unsqueeze(1)).sum(dim=3)  # (B, max_n, 6)
        #        
        #        best_idx = th.argmin(distances, dim=2)  # shape: (B, max_n)
        #        valid_idx = th.nonzero(valid_mask, as_tuple=False)  # shape: (total_valid, 2)
     #
        #        best_candidate_indices = best_idx[valid_mask]  # shape: (total_valid,)
        #        result = batch.t[valid_idx[:, 0], best_candidate_indices]  # shape: (total_valid, H)
        #        return self.loss(comp, result).unsqueeze(dim=0)
        if self.training:
            #t_flat = batch.t.view(batch.t.shape[0], -1)  
            #targets = t_flat.repeat_interleave(batch.n, dim=0)  
            return self.loss(comp, batch.t).unsqueeze(dim=0) #self.loss(comp, targets).unsqueeze(dim=0)
        else :
            return comp


