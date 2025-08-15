
import typing
import multiprocessing as mp
import multiprocessing.shared_memory

import numpy as np
import torch as th
import h5py

from dataclasses import dataclass

# The size of positional embedding for solutions graph
PARAM_UI_DIM =  8      #
PARAM_UP_DIM =  8      #
PARAM_VI_DIM =  6      #
PARAM_VP_DIM = 64      #
PARAM_NEDGES_TYPE = 19  #


VRP_MAX_NODES        = 1250 # Maximal amount of nodes allowed in a VRP
VRP_MAX_DUMMY_DEPOTS = 128  # Maximal amount of dummy depots allowed in a VRP
VRP_MAX_SUBGRAPHS    = 32   # Maximal amount of subgraphs in one sample
VRP_MAX_SAMPLE_SSIZE = 1024 # The maximal amount of solutions in all subgraphs of a (training) sample


# Normal distribution of subgraphs lenght in samples
PARAM_SUBGRAPH_SAMPLE_MU = 5.
PARAM_SUBGRAPH_SAMPLE_SIGMA = 2.

# It stores samples for ML model
# The fields with asterix (*) are copied from samplers into the trainer
@dataclass
class Sample :
    n : typing.Union[int, np.ndarray, th.LongTensor]         # (*) Number of vertices in solutions graph
    m : typing.Union[int, np.ndarray, th.LongTensor]         # (*) Number of edges in solutions graph
    l : typing.Union[int, np.ndarray, th.LongTensor]         # (*) Number of local vectors in the solution
    s : int         # Number of possible moves to connect solutions
    u_dim : int     # Dimension of local VRP embeddings
    v_dim : int     # Dimension of global VRP embeddings
    u : typing.Union[np.ndarray, th.FloatTensor]   # (*) Global embeddigns n x g
    v : typing.Union[np.ndarray, th.FloatTensor]   # (*) Vertices embeddings n * l x h
    e : typing.Union[np.ndarray, th.LongTensor]    # (*) Edges embeddings m x 3
    # ------------------------------------
    i_ptr : typing.Optional[np.ndarray]   # neighbors list begin
    n_arr : typing.Optional[np.ndarray]   # neighbors list indices
    g_dst : typing.Union[np.ndarray, th.LongTensor]   # (*) distance to solution
    m_bst : typing.Union[np.ndarray, th.FloatTensor]  # (*) the best move from the vertice
    t :  typing.Optional[ typing.Union[np.ndarray, th.FloatTensor] ] = None # A ground truth for E2E training or array of routes for inference


# Save to hdf5
def save_samples_hdf5(samples : typing.List[Sample], file_name : str) -> None:
    """Saves a list of Sample objects to an HDF5 file."""
    with h5py.File(file_name, 'w') as f:
        for i, sample in enumerate(samples):
            # Create a group for each sample
            grp = f.create_group(f"sample_{i}")
            
            # Save scalar attributes
            grp.attrs['n'] = sample.n
            grp.attrs['m'] = sample.m
            grp.attrs['s'] = sample.s
            grp.attrs['l'] = sample.l
            grp.attrs['u_dim'] = sample.u_dim
            grp.attrs['v_dim'] = sample.v_dim

            # Save tensors and arrays as datasets
            grp.create_dataset('u', data=sample.u.numpy() if isinstance(sample.u, th.Tensor) else sample.u)
            grp.create_dataset('v', data=sample.v.numpy() if isinstance(sample.v, th.Tensor) else sample.v)

            if sample.e is not None:
                grp.create_dataset('e', data=sample.e.numpy() if isinstance(sample.e, th.Tensor) else sample.e)
            if sample.i_ptr is not None:
                grp.create_dataset('i_ptr', data=sample.i_ptr)
            if sample.n_arr is not None:
                grp.create_dataset('n_arr', data=sample.n_arr)
            if sample.g_dst is not None:
                grp.create_dataset('g_dst', data=sample.g_dst)
            if sample.m_bst is not None:
                grp.create_dataset('m_bst', data=sample.m_bst)
            if sample.t is not None:
                grp.create_dataset('t', data=sample.t)


# Load from hdf5
def load_samples_hdf5(file_name : str) -> typing.List[Sample] :
    """Loads a list of Sample objects from an HDF5 file."""
    samples = []
    with h5py.File(file_name, 'r') as f:
        for key in f:
            grp = f[key]
            # Reconstruct the Sample object
            sample = Sample(
                n=grp.attrs['n'],
                m=grp.attrs['m'],
                s=grp.attrs['s'],
                l=grp.attrs['l'],
                u_dim=grp.attrs['u_dim'],
                v_dim=grp.attrs['v_dim'],
                u=grp['u'][:],
                v=grp['v'][:],
                e=grp['e'][:],
                i_ptr=grp['i_ptr'][:] if 'indptr' in grp else None,
                n_arr=grp['n_arr'][:] if 'n_arr' in grp else None,
                g_dst=grp['g_dst'][:] if 'g_dst' in grp else None,
                m_bst=grp['m_bst'][:] if 'm_bst' in grp else None,
                t = grp['t'][:] if 't' in grp else None
            )
            samples.append(sample)
    return samples



