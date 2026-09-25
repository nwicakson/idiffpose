# models/deq_wrapper.py
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn, autograd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Shared stats holder (optional)
# -----------------------------------------------------------------------------
@dataclass
class _DeqStats:
    f_residuals: List[float] = field(default_factory=list)
    b_residuals: List[float] = field(default_factory=list)
    used_iters_f: int = 0
    used_iters_b: int = 0

LAST_DEQ_STATS = _DeqStats()

# -----------------------------------------------------------------------------
# Safe utilities
# -----------------------------------------------------------------------------
def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Finite norm with nan_to_num guard
    n = torch.linalg.vector_norm(x.reshape(x.shape[0], -1), ord=2, dim=1)  # [B]
    n = torch.nan_to_num(n, nan=math.inf, posinf=math.inf, neginf=math.inf)
    return torch.clamp(n, min=eps)

def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _replace_nonfinite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------------------------------------------------------
# Anderson solver (robust)
# -----------------------------------------------------------------------------
def anderson(
    f: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    m: int = 5,
    lam: float = 1e-4,
    max_iter: int = 25,
    tol: float = 1e-4,
    beta: float = 0.7,
    damping: float = 0.5,
    tag: str = "F",
    track_residuals: bool = False,
    max_residual: float = 1e6,
) -> Tuple[torch.Tensor, int, List[float]]:
    """
    Damped Anderson acceleration with safety guards.
    z_{k+1} = (1 - damping) * z_k + damping * (f(z_k) + mix)
    """
    B = z0.shape[0]
    z = z0
    fz = f(z)
    z = _replace_nonfinite(z)
    fz = _replace_nonfinite(fz)

    X = []
    F = []
    res_hist: List[float] = []

    def _residual(z, fz) -> torch.Tensor:
        r = (fz - z)
        n = _safe_norm(r)  # [B]
        # batch reduce to scalar to guide stopping (mean over batch)
        val = torch.mean(torch.clamp(n, max=max_residual)).item()
        return val

    best = z
    best_res = float("inf")

    for k in range(max_iter):
        res_val = _residual(z, fz)
        if track_residuals:
            res_hist.append(res_val)
        LAST_DEQ_STATS.used_iters_f = k + 1

        # stop if good enough
        if res_val < tol:
            break

        if not math.isfinite(res_val) or res_val >= max_residual:
            # Bail out: return last best
            logger.warning(f"[DEQ-{tag}] residual invalid/too large at iter {k}: {res_val:.3e}. "
                           f"Returning last good iterate.")
            return best.detach(), k + 1, res_hist

        # record best
        if res_val < best_res:
            best = z.detach()
            best_res = res_val

        # build history
        X.append(z)
        F.append(fz)
        if len(X) > m:
            X.pop(0)
            F.pop(0)

        # solve small least squares for alpha
        # (I + lam*I) G G^T alpha = ones, where G = [F_i - X_i]
        G = torch.stack([f - x for (x, f) in zip(X, F)], dim=1)  # [B, k, ...]
        # Flatten features
        Bk = G.shape[1]
        Gf = G.reshape(B, Bk, -1)  # [B, k, D]
        GT = torch.transpose(Gf, 1, 2)  # [B, D, k]
        # (G G^T + lam I) a = 1
        GGt = torch.bmm(Gf, GT)  # [B, k, k]
        eye = torch.eye(Bk, device=GGt.device, dtype=GGt.dtype).unsqueeze(0).expand_as(GGt)
        GGt_damped = GGt + lam * eye
        ones = torch.ones((B, Bk, 1), device=GGt.device, dtype=GGt.dtype)

        try:
            alpha = torch.linalg.solve(GGt_damped, ones)  # [B, k, 1]
        except RuntimeError:
            # fallback: least squares
            alpha = torch.linalg.lstsq(GGt_damped, ones).solution
        # normalize alphas
        alpha = alpha / torch.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8)

        # compute Anderson step: z + sum_i alpha_i (F_i - X_i)
        step = torch.bmm(Gf.transpose(1, 2), alpha).squeeze(-1)  # [B, D]
        step = step.view_as(z)

        # mix (beta) and damping (line search)
        z_new = (1 - damping) * z + damping * (fz + beta * (step))
        z_new = _replace_nonfinite(z_new)

        z = z_new
        fz = f(z)
        z = _replace_nonfinite(z)
        fz = _replace_nonfinite(fz)

    return z.detach(), LAST_DEQ_STATS.used_iters_f, res_hist

# -----------------------------------------------------------------------------
# Autograd wrapper for implicit DEQ
# -----------------------------------------------------------------------------
class _DEQFixedPoint(autograd.Function):
    @staticmethod
    def forward(ctx,
                z0: torch.Tensor,
                cell: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                cfg: dict,
                mask: torch.Tensor,
                temb: torch.Tensor):
        """
        z* = fixed point of z = cell(z, mask, temb)
        """
        mode_block = cfg.get("mode_block", "implicit")
        iterations = int(cfg.get("iterations", 15))
        track_residuals = bool(cfg.get("track_residuals", False))

        # implicit (Anderson) vs unrolled
        if mode_block == "implicit":
            z_star, iters_f, res_hist_f = anderson(
                f=lambda z: cell(z, mask, temb),
                z0=z0,
                m=int(cfg.get("anderson_m", 5)),
                lam=float(cfg.get("anderson_l2", 1e-4)),
                max_iter=int(cfg.get("solver_max_iter", iterations)),
                tol=float(cfg.get("solver_tol", 1e-4)),
                beta=float(cfg.get("anderson_beta", 0.7)),
                damping=float(cfg.get("solver_damping", 0.5)),
                tag="F",
                track_residuals=track_residuals,
                max_residual=float(cfg.get("max_residual", 1e6)),
            )
            LAST_DEQ_STATS.f_residuals = res_hist_f
            used_iters = iters_f
        else:
            # Unrolled fixed K steps (explicit)
            z = z0
            used_iters = 0
            for k in range(iterations):
                used_iters += 1
                z_next = cell(z, mask, temb)
                if track_residuals:
                    res_k = torch.mean(_safe_norm(z_next - z)).item()
                    LAST_DEQ_STATS.f_residuals.append(res_k)
                z = z_next
            z_star = z.detach()

        # Save for backward
        ctx.cell = cell
        ctx.cfg = cfg
        ctx.mask = mask
        ctx.temb = temb
        ctx.save_for_backward(z_star)

        return z_star

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (z_star,) = ctx.saved_tensors
        cell = ctx.cell
        cfg = ctx.cfg
        mask = ctx.mask
        temb = ctx.temb

        mode_block = cfg.get("mode_block", "implicit")
        iterations = int(cfg.get("iterations", 15))
        track_residuals = bool(cfg.get("track_residuals", False))

        # Define linear map v -> J_f(z*)^T v where f(z) = cell(z)
        def JT_v(v):
            with torch.enable_grad():
                z_star_detached = z_star.detach().requires_grad_(True)
                f_z = cell(z_star_detached, mask, temb)
                (JTv,) = autograd.grad(f_z, z_star_detached, v, retain_graph=True, allow_unused=False)
            return JTv

        if mode_block == "implicit":
            # Solve (I - J_f(z*)^T) g = grad_output   via fixed-point: g = JT_v(g) + grad_output
            def g_map(g):
                return JT_v(g) + grad_output

            g, iters_b, res_hist_b = anderson(
                f=g_map,
                z0=torch.zeros_like(z_star),
                m=int(cfg.get("anderson_m", 5)),
                lam=float(cfg.get("anderson_l2", 1e-4)),
                max_iter=int(cfg.get("solver_max_iter", iterations)),
                tol=float(cfg.get("solver_tol", 1e-4)),
                beta=float(cfg.get("anderson_beta", 0.7)),
                damping=float(cfg.get("solver_damping", 0.5)),
                tag="B",
                track_residuals=track_residuals,
                max_residual=float(cfg.get("max_residual", 1e6)),
            )
            LAST_DEQ_STATS.b_residuals = res_hist_b
            LAST_DEQ_STATS.used_iters_b = iters_b
        else:
            # Unrolled backward with K steps (truncated backprop)
            g = grad_output
            for _ in range(iterations):
                g = JT_v(g) + grad_output

        # Gradient only w.r.t. first arg (z0) — others are non-tensor/ignored
        return g, None, None, None, None

# -----------------------------------------------------------------------------
# Block wrapper
# -----------------------------------------------------------------------------
class DEQAttentionBlock(nn.Module):
    """
    Wraps an Attn→GCN cell as either:
      - true DEQ (implicit, Anderson)
      - unrolled K-step cell
    """
    def __init__(self, atten_layer: nn.Module, gcn_layer: nn.Module,
                 name: str, config):
        super().__init__()
        self.name = name
        self.attn = atten_layer
        self.gcn = gcn_layer

        # read config
        deq_cfg = getattr(config, "deq", None)
        self.cfg = {
            "mode_block": getattr(deq_cfg, "mode_block", "implicit"),
            "iterations": int(getattr(deq_cfg, "iterations", 15)),
            "solver_max_iter": int(getattr(deq_cfg, "solver_max_iter", 25)),
            "solver_tol": float(getattr(deq_cfg, "solver_tol", 1e-4)),
            "anderson_m": int(getattr(deq_cfg, "anderson_m", 5)),
            "anderson_l2": float(getattr(deq_cfg, "anderson_l2", 1e-4)),
            "anderson_beta": float(getattr(deq_cfg, "anderson_beta", 0.7)),
            "solver_damping": float(getattr(deq_cfg, "solver_damping", 0.5)),
            "track_residuals": bool(getattr(deq_cfg, "track_residuals", False)),
            "max_residual": float(getattr(deq_cfg, "max_residual", 1e6)),
        }

        # runtime knobs updated by the runner (iterations / tolerance)
        self.iterations = self.cfg["iterations"]
        self.solver_tol = self.cfg["solver_tol"]

    def _cell(self, z: torch.Tensor, mask: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # One Attn → GCN step with time embedding injection
        z2 = self.attn(z, mask)
        z3 = self.gcn(z2, temb)
        return z3

    def forward(self, z0: torch.Tensor, mask: torch.Tensor, temb: torch.Tensor):
        # refresh runtime knobs
        cfg = dict(self.cfg)
        cfg["iterations"] = getattr(self, "iterations", cfg["iterations"])
        cfg["solver_tol"] = getattr(self, "solver_tol", cfg["solver_tol"])

        mode_block = cfg.get("mode_block", "implicit")
        if mode_block == "implicit":
            z_star = _DEQFixedPoint.apply(z0, self._cell, cfg, mask, temb)
            iters = LAST_DEQ_STATS.used_iters_f
            return z_star, iters
        else:
            # Unrolled
            z = z0
            for _ in range(int(cfg["iterations"])):
                z = self._cell(z, mask, temb)
            return z, int(cfg["iterations"])

# -----------------------------------------------------------------------------
# Manager (keeps per-block runtime stats & access)
# -----------------------------------------------------------------------------
class DEQManager:
    def __init__(self, config):
        self.components: List[DEQAttentionBlock] = []
        self.config = config

    def register(self, comp: DEQAttentionBlock):
        self.components.append(comp)

    def update_stats(self, iters: int):
        # placeholder if you want to aggregate per-iter stats
        pass
