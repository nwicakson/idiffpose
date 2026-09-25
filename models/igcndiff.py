# models/igcndiff.py
from __future__ import absolute_import

import math
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.GraFormer import MultiHeadedAttention, GraAttenLayer, GraphNet
from models.deq_wrapper import DEQAttentionBlock, DEQManager

__all__ = ["GCNdiff", "adj_mx_from_edges"]

# ---------------------------------------------------------------------
# Adjacency utility (needed by runners.idiffpose_frame import)
# ---------------------------------------------------------------------
def adj_mx_from_edges(num_pts: int,
                      edges: torch.Tensor,
                      sparse: bool = False,
                      self_connections: bool = True,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build a symmetric adjacency matrix from an edge list.

    Args:
        num_pts: number of graph nodes (joints)
        edges:   LongTensor of shape [E, 2] with 0-based (i,j) pairs
        sparse:  if True returns torch.sparse_coo_tensor, else dense Tensor
        self_connections: add self loops on the diagonal
        dtype:   dtype of returned tensor

    Returns:
        A (num_pts x num_pts) adjacency (dense or sparse)
    """
    if not torch.is_tensor(edges):
        edges = torch.tensor(edges, dtype=torch.long)
    assert edges.ndim == 2 and edges.size(1) == 2, "edges must be shape [E,2]"

    i = edges[:, 0]
    j = edges[:, 1]

    # undirected
    idx_i = torch.cat([i, j], dim=0)
    idx_j = torch.cat([j, i], dim=0)

    if self_connections:
        loop_idx = torch.arange(num_pts, dtype=torch.long)
        idx_i = torch.cat([idx_i, loop_idx], dim=0)
        idx_j = torch.cat([idx_j, loop_idx], dim=0)

    vals = torch.ones(idx_i.numel(), dtype=dtype)
    indices = torch.stack([idx_i, idx_j], dim=0)  # [2, E*2(+loops)]

    if sparse:
        adj = torch.sparse_coo_tensor(indices, vals, size=(num_pts, num_pts), dtype=dtype)
        # make sure coalesced
        adj = adj.coalesce()
        return adj
    else:
        adj = torch.zeros((num_pts, num_pts), dtype=dtype)
        adj[indices[0], indices[1]] = 1.0
        return adj

# ---------------------------------------------------------------------
# Diffusion timestep embedding
# ---------------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings (DDPM / Fairseq-style).
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

class _ResChebGC_diff(nn.Module):
    """
    Graph block with time embedding injection (used in diffusion denoiser).
    """
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        self.temb_proj = torch.nn.Linear(emd_dim, hid_dim)

    def forward(self, x, temb):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out

class GCNdiff(nn.Module):
    """
    DiffPose denoiser backbone with optional DEQ:
      - deq.mode == "per_layer": wrap selected layer indices as DEQ blocks
      - deq.mode == "stack":     build a fresh weight-tied Attn→GCN cell and wrap
                                 the entire trunk (between gconv_input and gconv_output)
    """
    def __init__(self, adj, config):
        super(GCNdiff, self).__init__()

        self.adj = adj
        self.config = config

        # --- Load model config ---
        con_gcn = config.model
        (self.hid_dim, self.emd_dim, self.coords_dim,
         num_layers, n_head, dropout, n_pts) = (
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim,
            con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
        )
        # Always expand emd_dim to 4x hid (matches original implementation)
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim * 4
        self.n_layers = num_layers

        logging.info(f"GCNdiff | L={num_layers} | hid={self.hid_dim} | heads={n_head}")

        # --- Build GraphFormer trunk (explicit layers for per_layer mode) ---
        self.gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_layers = []
        _attention_layers = []

        dim_model = self.hid_dim
        c = copy.deepcopy
        base_attn = MultiHeadedAttention(n_head, dim_model)
        base_gcn  = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for _ in range(num_layers):
            _gconv_layers.append(
                _ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                                emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1)
            )
            _attention_layers.append(GraAttenLayer(dim_model, c(base_attn), c(base_gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layers)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)

        # --- Diffusion time embedding MLP ---
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim, self.emd_dim),
            torch.nn.Linear(self.emd_dim, self.emd_dim),
        ])

        # --- DEQ wiring ---
        self.deq_manager = DEQManager(config)
        self.deq_enabled = hasattr(config, 'deq') and getattr(config.deq, 'enabled', False)
        self.deq_mode    = getattr(config.deq, 'mode', 'per_layer')
        cfg_layers = getattr(config.deq, 'layers', [])
        self.deq_layers = list(cfg_layers) if isinstance(cfg_layers, (list, tuple)) else ([int(cfg_layers)] if cfg_layers != [] else [])

        # explicit block-level mode banner
        self.deq_mode_block = getattr(config.deq, 'mode_block', 'unrolled')
        if self.deq_enabled:
            if self.deq_mode_block == 'implicit':
                logging.info(
                    "DEQ block mode: implicit (true DEQ w/ Anderson). "
                    f"solver_max_iter={getattr(config.deq,'solver_max_iter',25)}, "
                    f"solver_tol={getattr(config.deq,'solver_tol',1e-4)}, "
                    f"anderson_m={getattr(config.deq,'anderson_m',5)}, "
                    f"anderson_l2={getattr(config.deq,'anderson_l2',1e-4)}"
                )
            else:
                logging.info(
                    "DEQ block mode: unrolled (explicit K steps). "
                    f"default_iterations={getattr(config.deq,'default_iterations', getattr(config.deq,'iterations',15))}, "
                    f"best_epoch_iterations={getattr(config.deq,'best_epoch_iterations', None)}"
                )

        # Build per-layer or stack DEQ
        if self.deq_enabled:
            if self.deq_mode == 'per_layer':
                # make this a ModuleDict so components are visible in state_dict (helps some checkpoint namespaces)
                self.per_layer_deq_blocks = nn.ModuleDict()
                valid = []
                for idx in self.deq_layers:
                    if 0 <= idx < self.n_layers:
                        block = DEQAttentionBlock(self.atten_layers[idx],
                                                  self.gconv_layers[idx],
                                                  name=f"layer_{idx}", config=config)
                        self.per_layer_deq_blocks[f"layer_{idx}"] = block
                        self.deq_manager.register(block)
                        valid.append(idx)
                    else:
                        logging.warning(f"DEQ(per_layer): index {idx} out of range [0,{self.n_layers-1}] (skipped)")
                logging.info(f"DEQ per-layer mode: wrapping layers {sorted(valid)}")

            elif self.deq_mode == 'stack':
                # Ignore deq.layers in stack mode; build a fresh weight-tied cell
                if len(self.deq_layers) > 0:
                    logging.warning(f"DEQ stack mode: deq.layers={self.deq_layers} specified but ignored.")
                tied_attn = GraAttenLayer(dim_model, c(base_attn), c(base_gcn), dropout)
                tied_gcn  = _ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                                            emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1)
                self.deq_stack_block = DEQAttentionBlock(tied_attn, tied_gcn, name="trunk", config=config)
                self.deq_manager.register(self.deq_stack_block)
                logging.info("DEQ stack mode: monolithic trunk enabled.")

            else:
                logging.warning(f"Unknown deq.mode='{self.deq_mode}', running without DEQ.")
                self.deq_enabled = False

            # Make iterations/tolerance robust to YAML naming
            iters = getattr(config.deq, 'iterations',
                            getattr(config.deq, 'default_iterations', None))
            tol   = getattr(config.deq, 'tolerance', None)
            if iters is not None or tol is not None:
                for comp in self.deq_manager.components:
                    if iters is not None:
                        comp.iterations = iters
                    if tol is not None:
                        # used by implicit Anderson solver
                        comp.solver_tol = tol

    def forward(self, x, mask, t, cemd):
        # timestep embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # input projection
        out = self.gconv_input(x, self.adj)

        if not self.deq_enabled:
            # Original explicit trunk
            for i in range(self.n_layers):
                out = self.atten_layers[i](out, mask)
                out = self.gconv_layers[i](out, temb)

        else:
            if self.deq_mode == 'per_layer':
                for i in range(self.n_layers):
                    key = f"layer_{i}"
                    if hasattr(self, "per_layer_deq_blocks") and (key in self.per_layer_deq_blocks):
                        out, iters = self.per_layer_deq_blocks[key](out, mask, temb)
                        self.deq_manager.update_stats(iters)
                        if getattr(self.config.deq, 'track_residuals', False):
                            from models.deq_wrapper import LAST_DEQ_STATS
                            if self.deq_mode_block == 'implicit':
                                final_res = (LAST_DEQ_STATS.f_residuals[-1] if LAST_DEQ_STATS.f_residuals else float('nan'))
                                logging.info(f"[DEQ layer {i}] implicit: used_iters={iters}, final_rel_res={final_res:.3e}")
                            else:
                                logging.info(f"[DEQ layer {i}] unrolled: used_iters={iters}")
                    else:
                        out = self.atten_layers[i](out, mask)
                        out = self.gconv_layers[i](out, temb)

            elif self.deq_mode == 'stack':
                # Single equilibrium trunk between input/output
                out, iters = self.deq_stack_block(out, mask, temb)
                self.deq_manager.update_stats(iters)
                if getattr(self.config.deq, 'track_residuals', False):
                    from models.deq_wrapper import LAST_DEQ_STATS
                    if self.deq_mode_block == 'implicit':
                        final_res = (LAST_DEQ_STATS.f_residuals[-1] if LAST_DEQ_STATS.f_residuals else float('nan'))
                        logging.info(f"[DEQ stack] implicit: used_iters={iters}, final_rel_res={final_res:.3e}")
                    else:
                        logging.info(f"[DEQ stack] unrolled: used_iters={iters}")

            else:
                # Fallback to explicit if mode unknown
                for i in range(self.n_layers):
                    out = self.atten_layers[i](out, mask)
                    out = self.gconv_layers[i](out, temb)

        # output projection
        out = self.gconv_output(out, self.adj)
        return out
