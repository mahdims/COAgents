import typing
import numpy as np
import torch as th

from collections import OrderedDict

# The loss layer
class LossBCELayer(th.nn.Module) :
    def __init__(self,
                 nedges_types : int, # maximum of edges types
                 ez_dim: int,        # number of edge embeddings
                 u_dim  : int,       # input dimension
                 v_dim  : int,       # position embedding dimension
                 z_dim  : int        # inner embedding space
                 ) -> None :
        super(LossBCELayer, self).__init__()
        self.u_dim  = u_dim
        self.v_dim  = v_dim
        self.z_dim  = z_dim
        self.ez_dim = ez_dim
        self.nedges_types = nedges_types

        # Convolution
        self.G = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear((u_dim + v_dim), 2 * (u_dim + v_dim + z_dim), bias=True, device=th.device('cpu'), dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * (u_dim + v_dim + z_dim), z_dim, bias=True, device=th.device('cpu'), dtype=th.float32)),
            ('gelu2',    th.nn.GELU())
        ]))
        # Envinromental part
        self.ZV = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear(z_dim, 2 * z_dim, bias=True, device=th.device('cpu'), dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * z_dim,   1, bias=True, device=th.device('cpu'), dtype=th.float32)),
            ('sigmoid2', th.nn.Sigmoid())
        ]))
        # Classifiers
        self.ZE  = th.nn.Sequential(OrderedDict([
            ('linear1',  th.nn.Linear(z_dim + ez_dim, 2 * z_dim, bias=True, device=th.device('cpu'), dtype=th.float32)),
            ('relu1',    th.nn.ReLU()),
            ('linear2',  th.nn.Linear(2 * z_dim,   1, bias=True, device=th.device('cpu'), dtype=th.float32)),
            ('sigmoid2', th.nn.Sigmoid())
        ]))
        th.nn.init.xavier_uniform_(self.G[0].weight.data)  ; th.nn.init.zeros_(self.G[0].bias)
        th.nn.init.xavier_uniform_(self.G[2].weight.data)  ; th.nn.init.zeros_(self.G[2].bias)
        th.nn.init.xavier_uniform_(self.ZV[0].weight.data) ; th.nn.init.zeros_(self.ZV[0].bias)
        th.nn.init.xavier_uniform_(self.ZV[2].weight.data) ; th.nn.init.zeros_(self.ZV[2].bias)
        th.nn.init.xavier_uniform_(self.ZE[0].weight.data) ; th.nn.init.zeros_(self.ZE[0].bias)
        th.nn.init.xavier_uniform_(self.ZE[2].weight.data) ; th.nn.init.zeros_(self.ZE[2].bias)

        # Cross-entropy loss
        self.loss_fx = th.nn.BCELoss(reduction='none')

    # The forward
    def forward (self,
                 n  : th.LongTensor,      # Size of samples blocks
                 l  : th.LongTensor,      # Sizes of solutions blocks
                 u  : th.FloatTensor,     # Input feature embedding tensor
                 v  : th.FloatTensor,     # Input position embeding tensor
                 sc_indx : th.LongTensor, # Indices for in-solution convolution (for speedup)
                 label_solutions  : typing.Optional[th.LongTensor] = None, # labels for solutions to be locally optimal
                 label_heuristics : typing.Optional[th.LongTensor] = None, # labels for edges to be locally optimal
                 ee : typing.Optional[th.FloatTensor] = None,              # Embedding of ALL moves [nedges_types, ez_dim]
                 debug_layers : typing.Optional[typing.List[typing.Tuple[ str, th.Tensor ]]] = None,
                 ) -> typing.Union[ th.FloatTensor, th.LongTensor ] :  # graph nodes embeddings and graph edges embeddings
        if label_solutions  is not None :
            assert label_solutions.nelement() == n.sum() , "Sizes of solutions labels doesn't match solutions number"
        if label_heuristics is not None :
            assert label_heuristics.nelement() == u.shape[0] * self.nedges_types , "Sizes of labels and representations doesn't match!"

        # Convolve the solutions in blocks 1D convolution
        s_conv = self.G.forward(th.cat((u.repeat_interleave(l[0].item(), 0), v), dim=1))
        #if sc_indx is None : sc_indx = th.arange(l.shape[0], dtype=th.long, device=l.device).repeat_interleave(l, 0)
        z = th.zeros(n.sum(), s_conv.shape[1], dtype=v.dtype, device=v.device).index_add(0, sc_indx, s_conv) / l[0].item()
        # Debugging
        #debug_layers.append(('z', z)) ; z.retain_grad()

        # Trigger for executiojn mode: inference vs training
        if self.training : # Compute BCE loss
            # Declare global loss
            loss_total = th.zeros(1, 1, dtype=th.float32, device=z.device, requires_grad=True)
            # Apply BCE loss to nodes
            if label_solutions is not None :
                # Classify vertices
                if label_solutions is not None:
                    zv = self.ZV.forward(z)
                    # Debugging
                    # debug_layers.append(('zv', zv)) ; zv.retain_grad()
                # Evaluate imbalanced loss
                loss_solutions  = self.loss_fx(zv.squeeze(), label_solutions)
                # Compute nodes weights
                indx_solutions = th.arange(n.nelement(), dtype=th.long, device=n.device, requires_grad=False).repeat_interleave(n, dim=0)
                label_solutions_sum   = th.zeros(n.nelement(), dtype=label_solutions.dtype, device=n.device).index_add(0, indx_solutions, label_solutions)
                weight_solutions_loss = ((n - label_solutions_sum) / label_solutions_sum).repeat_interleave(n)
                weight_solutions_loss = th.ones_like(loss_solutions) + label_solutions * (weight_solutions_loss - 1.)
                # Update the total loss
                loss_total = loss_total + (weight_solutions_loss * loss_solutions).sum(dim=0, keepdim=True)
                # Put the extra-scaler for solutions loss which is useful if heuristics loss is also enabled
                loss_total = loss_total * float(self.nedges_types)
            # Apply BCE loss to heuristics
            if label_heuristics is not None:
                # Classify moves
                if label_heuristics is not None:
                    ze = self.ZE.forward(th.cat((z.repeat_interleave(self.nedges_types, dim=0),
                                        (ee.flatten()).repeat(n.sum()).view(n.sum() * self.nedges_types, ee.shape[1])), dim=1))
                    # Debugging
                    #debug_layers.append(('ze', ze)) ; ze.retain_grad()
                # Evaluate imbalanced loss
                loss_heuristics = self.loss_fx(ze.squeeze(), label_heuristics)
                # Compute edge weights
                indx_heuristics = th.arange(n.nelement(), dtype=th.long, device=n.device, requires_grad=False).repeat_interleave(n * self.nedges_types, dim=0)
                label_heuristics_sum   = th.zeros(n.nelement(), dtype=label_heuristics.dtype, device=n.device).index_add(0, indx_heuristics, label_heuristics)
                weight_heuristics_loss = ((n * self.nedges_types - label_heuristics_sum) / label_heuristics_sum).repeat_interleave(n * self.nedges_types)
                weight_heuristics_loss = th.ones_like(loss_heuristics) + label_heuristics * (weight_heuristics_loss - 1.)
                # Update the total loss
                loss_total = loss_total + (weight_heuristics_loss * loss_heuristics).sum(dim=0, keepdim=True)
        else             : # Compute amount of classification matches
            # Apply BCE logit to solutions
            loss_solutions = None
            if label_solutions is not None :
                # Classify vertices
                if label_solutions is not None :
                    zv = self.ZV.forward(z)
                s_y = (zv.squeeze() >= 0.5).to(dtype=th.bool) ; s_l = label_solutions.to(dtype=th.bool)
                loss_solutions = th.concatenate(( (( s_y) & ( s_l)).sum(dim=0, keepdim=True, dtype=th.long),  # t_positives
                                                  (( s_y) & (~s_l)).sum(dim=0, keepdim=True, dtype=th.long),  # f_positives
                                                  ((~s_y) & (~s_l)).sum(dim=0, keepdim=True, dtype=th.long),  # t_negatives
                                                  ((~s_y) & ( s_l)).sum(dim=0, keepdim=True, dtype=th.long),  # f_negatives
                                                 ), dim=0)
            # Apply BCE logit to heuristics
            loss_heuristics = None
            if label_heuristics is not None :
                # Classify moves
                if label_heuristics is not None :
                    ze = self.ZE.forward(th.cat((z.repeat_interleave(self.nedges_types, dim=0),
                                        (ee.flatten()).repeat(n.sum()).view(n.sum() * self.nedges_types, ee.shape[1])), dim=1))
                h_y = (ze.squeeze() >= 0.5).to(dtype=th.bool) ; h_l = label_heuristics.to(dtype=th.bool)
                loss_heuristics = th.concatenate(( (( h_y) & ( h_l)).sum(dim=0, keepdim=True, dtype=th.long),  # t_positives
                                                   (( h_y) & (~h_l)).sum(dim=0, keepdim=True, dtype=th.long),  # f_positives
                                                   ((~h_y) & (~h_l)).sum(dim=0, keepdim=True, dtype=th.long),  # t_negatives
                                                   ((~h_y) & ( h_l)).sum(dim=0, keepdim=True, dtype=th.long),  # f_negatives
                                                  ), dim=0)
            # Stich the losses
            if label_solutions is not None :
                if label_heuristics is not None :
                    loss_total = th.concatenate((loss_solutions, loss_heuristics), dim=0).unsqueeze(dim=0)
                else                            :
                    loss_total = loss_solutions.unsqueeze(dim=0)
            else                           :
                if label_heuristics is not None :
                    loss_total = loss_heuristics.unsqueeze(dim=0)
                else                            :
                    loss_total = th.concatenate( (self.ZV.forward(z).flatten(),
                                                  self.ZE.forward(th.cat((z.repeat_interleave(self.nedges_types, dim=0),
                                                 (ee.flatten()).repeat(n.sum()).view(n.sum() * self.nedges_types, ee.shape[1])), dim=1)).flatten()), dim=0)
        return loss_total
