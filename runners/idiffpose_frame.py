# runners/idiffpose_frame.py
import os
import logging
import time
import argparse
import os.path as path
import traceback

import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

# Models
from models.gcnpose import GCNpose, adj_mx_from_edges as adj_mx_from_edges_pose
from models.igcndiff import GCNdiff, adj_mx_from_edges as adj_mx_from_edges_diff
from models.ema import EMAHelper

# Utils
from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps
from common.data_utils import fetch_me, read_3d_data_me, create_2d_data
from common.generators import PoseGenerator_gmm
from common.loss import mpjpe, p_mpjpe

import time
from models.deq_wrapper import LAST_DEQ_STATS

# ----------------------------- helpers -----------------------------

def _normalize_checkpoint(ckpt):
    """
    Normalize a checkpoint (dict or list/tuple) into a canonical mapping:
      - 'model_state' : dict of parameter tensors (may be empty)
      - 'optimizer'   : optimizer state or None
      - 'epoch'       : epoch int or None
      - 'step'        : step int or None

    Handles:
      - dict with optional 'state_dict' key
      - list/tuple where element 0 is model state (or dict containing 'state_dict'),
        element 1 optimizer, element 2 epoch, element 3 step (common 'states' format)
    """
    model_state = {}
    opt_state = None
    epoch = None
    step = None

    # list/tuple checkpoint (runner saved "states" format)
    if isinstance(ckpt, (list, tuple)):
        first = ckpt[0] if len(ckpt) > 0 else None
        second = ckpt[1] if len(ckpt) > 1 else None
        third = ckpt[2] if len(ckpt) > 2 else None
        fourth = ckpt[3] if len(ckpt) > 3 else None

        if isinstance(first, dict):
            model_state = first.get("state_dict", first) if ("state_dict" in first) else first
        else:
            model_state = {}

        opt_state = second if isinstance(second, dict) else None
        epoch = int(third) if isinstance(third, (int, float)) else None
        step  = int(fourth) if isinstance(fourth, (int, float)) else None

    elif isinstance(ckpt, dict):
        # Dict could be raw state_dict or training container
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model_state = ckpt["state_dict"]
        else:
            # If values look tensor-like, assume raw model_state
            tensor_like = any(hasattr(v, "shape") for v in ckpt.values())
            if tensor_like and len(ckpt) > 0:
                model_state = ckpt
            else:
                model_state = ckpt.get("model_state", ckpt.get("state_dict", {}))
                opt_state = ckpt.get("optimizer", None)
                epoch = ckpt.get("epoch", None)
                step = ckpt.get("step", None)
    else:
        model_state = {}

    if not isinstance(model_state, dict):
        model_state = {}

    return {"model_state": model_state, "optimizer": opt_state, "epoch": epoch, "step": step}


def _torch_load_safe(ckpt_path: str):
    """torch.load with weights_only when available (PyTorch ≥2.4), else fallback."""
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def _infer_base(model_sd: dict, token: str):
    """Find the prefix in model_sd that contains '.{token}.' and return 'prefix.{token}.'"""
    marker = f".{token}."
    for k in model_sd.keys():
        if marker in k:
            return k.split(marker)[0] + marker
    return None

def _remap_stack_init(sd: dict, model_sd: dict) -> dict:
    """
    Remap keys from a stack-init file to whatever nesting your current model uses.
    Handles attention_layer, gcn_layer, temb.*, atten_layers/gconv_layers, and gconv1/2 vs g1/g2.
    Keeps only keys that both exist and shape-match.
    """
    def infer_base(token: str):
        marker = f".{token}."
        for k in model_sd.keys():
            if marker in k:
                return k.split(marker)[0] + marker
        return None

    base_attn = infer_base("attention_layer") or infer_base("atten_layer") or infer_base("attn")
    base_gcn  = infer_base("gcn_layer")       or infer_base("gconv_layer") or infer_base("gcn")

    # Try to find a temb path (either temb_mlp or temb.dense style)
    base_temb = None
    for k in model_sd.keys():
        if ".temb_mlp." in k:
            base_temb = k.split(".temb_mlp.")[0] + ".temb_mlp."
            break
        if ".temb." in k:
            base_temb = k.split(".temb.")[0] + ".temb."
            break

    sd_new = {}
    for k, v in sd.items():
        k2 = k

        # map stack-init namespaces to model namespaces
        if "attention_layer." in k and base_attn:
            k2 = base_attn + k.split("attention_layer.", 1)[1]
        elif "atten_layers." in k and base_attn:
            k2 = base_attn + k.split("atten_layers.", 1)[1]
        elif "atten_layer." in k and base_attn:
            k2 = base_attn + k.split("atten_layer.", 1)[1]
        elif "attn." in k and base_attn:
            k2 = base_attn + k.split("attn.", 1)[1]

        elif "gcn_layer." in k and base_gcn:
            k2 = base_gcn + k.split("gcn_layer.", 1)[1]
        elif "gconv_layers." in k and base_gcn:
            k2 = base_gcn + k.split("gconv_layers.", 1)[1]
        elif "gconv_layer." in k and base_gcn:
            k2 = base_gcn + k.split("gconv_layer.", 1)[1]
        elif ".gconv." in k and base_gcn:  # generic
            k2 = base_gcn + k.split(".gconv.", 1)[1]

        elif "temb_mlp." in k and base_temb:
            k2 = base_temb + k.split("temb_mlp.", 1)[1]
        elif "temb.dense." in k and base_temb:
            k2 = base_temb + k.split("temb.dense.", 1)[1]
        elif "temb_proj." in k and base_temb:
            # in some nets temb_proj.* maps under temb_mlp.* or temb.*
            suffix = k.split("temb_proj.", 1)[1]
            k2 = base_temb + suffix

        # Harmonize possible naming differences
        k2 = k2.replace(".gconv1.", ".g1.").replace(".gconv2.", ".g2.")

        if k2 in model_sd and model_sd[k2].shape == v.shape:
            sd_new[k2] = v
    return sd_new


# === Safe numeric helpers (prevents NaN scale during eval) ===
def _safe_scale(numerator: torch.Tensor, denominator: torch.Tensor, default: float = 1.0) -> torch.Tensor:
    """
    Returns numerator/denominator with NaN/Inf guarded.
    If denom is (near) zero, returns default. Works elementwise and preserves shape.
    """
    eps = 1e-8
    denom = torch.nan_to_num(denominator, nan=0.0, posinf=0.0, neginf=0.0)
    good = denom.abs() > eps
    out = torch.empty_like(denom)
    out[good] = numerator[good] / denom[good]
    out[~good] = default
    out = torch.nan_to_num(out, nan=default, posinf=default, neginf=default)
    return out


# ----------------------------- logging -----------------------------

_logger_initialized = False
def setup_logging():
    global _logger_initialized
    if _logger_initialized:
        return
    root_logger = logging.getLogger()
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    _logger_initialized = True

setup_logging()


# ============================= runner ==============================

class IDiffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.scaler = GradScaler()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model_var_type = config.model.var_type

        # GraFormer mask (CPU/GPU safe)
        self.src_mask = torch.tensor(
            [[[True]*17]]
        ).to(self.device)

        # Diffusion schedule
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # Best tracking
        self.best_mpjpe = float('inf')
        self.best_epoch = -1

        # DEQ schedule (single knob for both train & eval)
        self.deq_schedule = {
            'enabled': False,
            'iterations': 15,
            'tolerance': 0.001,
            'track_residuals': False
        }
        if hasattr(config, 'deq'):
            for key in self.deq_schedule.keys():
                if hasattr(config.deq, key):
                    self.deq_schedule[key] = getattr(config.deq, key)

        if hasattr(config, 'deq'):
            self.deq_schedule['enabled'] = config.deq.enabled
            logging.info(f"DEQ enabled status from config: {self.deq_schedule['enabled']}")

    # ------------------------- data prep --------------------------

    def prepare_data(self):
        args, config = self.args, self.config
        logging.info('==> Using settings {}'.format(args))

        if config.data.dataset == "human36m":
            logging.info('==> Loading Human36.6M dataset' if False else '==> Loading Human3.6M dataset')
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            logging.info('==> Dataset loaded, processing subjects')

            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS

            logging.info('==> Reading 3D data')
            self.dataset = read_3d_data_me(dataset)
            logging.info('==> Creating 2D data for training')
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            logging.info('==> Creating 2D data for testing')
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            logging.info('==> Setting up action filter')
            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = list(map(lambda x: dataset.define_actions(x)[0], self.action_filter))
                logging.info('==> Selected actions: {}'.format(self.action_filter))
            logging.info('==> Data preparation complete')
        else:
            raise KeyError('Invalid dataset')

        logging.info("Dataset information:")
        logging.info(f"Train subjects: {self.subjects_train}")
        logging.info(f"Test subjects: {self.subjects_test}")

    # ---------------------- deq iteration set ---------------------

    def _set_deq_iterations(self):
        if not self.deq_schedule['enabled']:
            logging.info("DEQ is disabled, using standard processing")
            return

        iterations = self.deq_schedule['iterations']
        if getattr(self.config.deq, 'enabled', False):
            mode_block = getattr(self.config.deq, 'mode_block', 'unrolled')
            if mode_block == 'implicit':
                logging.info("DEQ is enabled in IMPLICIT mode "
                            f"(solver_max_iter={getattr(self.config.deq,'solver_max_iter',25)}, "
                            f"solver_tol={getattr(self.config.deq,'solver_tol',1e-4)})")
            else:
                logging.info("DEQ is enabled in UNROLLED mode "
                            f"(default_iterations={getattr(self.config.deq,'default_iterations', getattr(self.config.deq,'iterations',15))}, "
                            f"best_epoch_iterations={getattr(self.config.deq,'best_epoch_iterations', None)})")

        if hasattr(self.model_diff, 'module') and hasattr(self.model_diff.module, 'deq_manager'):
            for comp in self.model_diff.module.deq_manager.components:
                prev = getattr(comp, 'iterations', None)
                comp.iterations = iterations
                if prev != iterations:
                    logging.info(f"Changed DEQ component '{comp.name}' from {prev} to {iterations} iterations")

        if hasattr(self.model_pose, 'module') and hasattr(self.model_pose.module, 'deq_manager'):
            for comp in self.model_pose.module.deq_manager.components:
                prev = getattr(comp, 'iterations', None)
                comp.iterations = iterations
                if prev != iterations:
                    logging.info(f"Changed DEQ component '{comp.name}' from {prev} to {iterations} iterations")

    # --------------------- model constructors ---------------------

    def create_diffusion_model(self, model_path=None):
        """
        Build diffusion model and optionally load pretrained weights.
        This loader accepts:
        - a raw state_dict (dict of param tensors)
        - a training 'states' list [state_dict, optimizer, epoch, step, ...]
        - a checkpoint saved as {'state_dict': ..., ...}
        - list/tuple wrappers (older checkpoints)
        It attempts several matching/remapping strategies to handle:
         - module. prefix differences
         - temb naming differences
         - gconv1/gconv2 vs g1/g2
         - stack-init vs per-layer DEQ namespaces (attention_layer/gcn_layer -> atten_layers/gconv_layers)
         - deq_block / deq_stack_block renames
        """
        config = self.config

        # Build adj for diffusion model (same joints edges you already had)
        edges = torch.tensor(
            [[0,1],[1,2],[2,3],
            [0,4],[4,5],[5,6],
            [0,7],[7,8],[8,9],[9,10],
            [8,11],[11,12],[12,13],
            [8,14],[14,15],[15,16]],
            dtype=torch.long
        )
        adj = adj_mx_from_edges_diff(num_pts=17, edges=edges, sparse=False)

        logging.info("Creating implicit diffusion model...")
        self.model_diff = GCNdiff(adj.to(self.device), config).to(self.device)
        self.model_diff = torch.nn.DataParallel(self.model_diff)

        # helper: get the raw dict contained in possible checkpoint formats
        def normalize_ckpt(ckpt):
            # ckpt may be: raw dict state_dict, dict with 'state_dict', list/tuple saved states, or old-style
            if ckpt is None:
                return None
            if isinstance(ckpt, dict):
                # if it's a wrapper with state_dict key
                if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                    return ckpt['state_dict']
                # sometimes training ckpt stored as {'model': state_dict}
                if 'model' in ckpt and isinstance(ckpt['model'], dict):
                    return ckpt['model']
                # raw dict of params
                return ckpt
            if isinstance(ckpt, (list, tuple)):
                # your saved 'states' often have [state_dict, optimizer, epoch, step, ...]
                # try to find the first element that's a dict-like
                for el in ckpt:
                    if isinstance(el, dict):
                        # if it looks like a state wrapper with 'state_dict'
                        if 'state_dict' in el and isinstance(el['state_dict'], dict):
                            return el['state_dict']
                        # otherwise assume el itself is the state_dict
                        return el
                # fallback: maybe first item is a dict state
                first = ckpt[0]
                if isinstance(first, dict):
                    return first
            # unknown type
            return None

        loaded_from_stack_init = False
        if model_path and os.path.isfile(model_path):
            logging.info(f"Loading pretrained IGCNdiff model from: {model_path}")
            raw = torch.load(model_path, map_location='cpu')

            sd_raw = normalize_ckpt(raw)
            if sd_raw is None:
                logging.warning("Checkpoint loaded but no state-dict-like object found; skipping weights load.")
            else:
                # model ref (unwrapped)
                model_ref = self.model_diff.module if isinstance(self.model_diff, torch.nn.DataParallel) else self.model_diff
                model_sd = model_ref.state_dict()
                total_params = len(model_sd)

                # small helpers
                def strip_module_prefix(sd_in):
                    return { (k[7:] if k.startswith('module.') else k): v for k, v in sd_in.items() }

                def add_module_prefix(sd_in):
                    new = {}
                    for k, v in sd_in.items():
                        if k.startswith('module.'):
                            new[k] = v
                        else:
                            new['module.' + k] = v
                    return new

                def swap_namespace(sd_in, a, b):
                    # produce new dict with a->b replacements on keys (if present)
                    out = {}
                    for k, v in sd_in.items():
                        if a in k:
                            out[k.replace(a, b, 1)] = v
                    return out

                # 0) attempt direct compatible keys (both raw and stripped)
                matched_total = 0
                loaded_keys = set()
                tried_variants = []

                # define candidate dicts to try (ordered)
                candidates = []
                # as-loaded
                candidates.append(("raw", sd_raw))
                # stripped module. prefix
                sd_stripped = strip_module_prefix(sd_raw)
                if sd_stripped != sd_raw:
                    candidates.append(("stripped_module", sd_stripped))
                # add module prefix (for the case model was in DataParallel)
                sd_with_module = add_module_prefix(sd_stripped)
                candidates.append(("added_module", sd_with_module))

                # remap stack-style namespaces into model namespaces (using helper remapper)
                try:
                    remapped_from_stripped = _remap_stack_init(sd_stripped, model_sd)
                    if remapped_from_stripped:
                        candidates.append(("remapped_from_stripped", remapped_from_stripped))
                except Exception:
                    pass
                try:
                    remapped_from_raw = _remap_stack_init(sd_raw, model_sd)
                    if remapped_from_raw:
                        candidates.append(("remapped_from_raw", remapped_from_raw))
                except Exception:
                    pass

                # heuristics for deq block naming swaps (some checkpoints use deq_block vs deq_stack_block)
                def add_deq_variants(sd_input):
                    variants = []
                    # deq_block -> deq_stack_block
                    v1 = swap_namespace(sd_input, "deq_block.", "deq_stack_block.")
                    if v1:
                        variants.append(("deq_block->deq_stack_block", v1))
                    v2 = swap_namespace(sd_input, "deq_stack_block.", "deq_block.")
                    if v2:
                        variants.append(("deq_stack_block->deq_block", v2))
                    # attention_layer.* -> atten_layers.* (strip index)
                    # (handled by _remap_stack_init, but keep small fallback)
                    v3 = {}
                    for k, vv in sd_input.items():
                        if "attention_layer." in k:
                            suffix = k.split("attention_layer.", 1)[1]
                            # try both with and without index
                            v3["atten_layers." + suffix] = vv
                    if v3:
                        variants.append(("attention_layer->atten_layers", v3))
                    return variants

                # append small deq variants from the raw loaded dict
                for name, base in list(candidates):
                    for hv_name, hv in add_deq_variants(base):
                        candidates.append((f"{name}+{hv_name}", hv))

                # keep track of keys we successfully loaded (avoid reloading same)
                matched_direct = 0
                for cand_name, cand_sd in candidates:
                    tried_variants.append(cand_name)
                    # choose only keys that match shapes in model_sd and not already loaded
                    compatible = {}
                    for k, v in cand_sd.items():
                        if k in model_sd and model_sd[k].shape == v.shape and k not in loaded_keys:
                            compatible[k] = v
                    if compatible:
                        try:
                            model_ref.load_state_dict(compatible, strict=False)
                        except Exception:
                            # best-effort: load partial via explicit param assignment
                            for k, v in compatible.items():
                                try:
                                    # direct assign to param.data if key exists
                                    param = dict(model_ref.named_parameters()).get(k, None)
                                    if param is not None:
                                        param.data.copy_(v)
                                except Exception:
                                    pass
                        # record loaded keys
                        for k in compatible.keys():
                            loaded_keys.add(k)
                        matched_direct += len(compatible)

                matched_union = len(loaded_keys)

                # Final reporting: compute missing & unexpected
                missing = sum(1 for k in model_sd.keys() if k not in loaded_keys)
                unexpected = sum(1 for k in sd_raw.keys() if ((k not in loaded_keys) and True))

                if matched_union == 0:
                    # helpful debug: dump a few example keys from both ckpt and model to inspect namespaces
                    logging.warning("Low direct match (0) and remap found no new keys. Consider dumping keys for debugging.")
                    ckpt_sample = list(sd_raw.keys())[:40]
                    model_sample = list(model_sd.keys())[:80]
                    logging.info("Sample checkpoint keys: " + ", ".join(ckpt_sample))
                    logging.info("Sample model keys: " + ", ".join(model_sample))
                    logging.info("Tried remap variants: " + ", ".join(tried_variants))

                logging.info(f"Loaded pretrained weights: matched (direct+remap) ~{matched_union}/{total_params} | missing≈{missing} | unexpected≈{unexpected}")
                loaded_from_stack_init = (matched_union > 0)

        else:
            logging.info("No --model_diff_path given or file not found; training diffusion model from scratch.")

        # Params info (post-load)
        total_params = sum(p.numel() for p in self.model_diff.parameters())
        trainable_params = sum(p.numel() for p in self.model_diff.parameters() if p.requires_grad)
        logging.info(f"IGCNdiff model created - Total parameters: {total_params}, Trainable: {trainable_params}")

        # If the user passed a training checkpoint that we didn't treat as stack-init, attempt fallback
        if model_path and (not loaded_from_stack_init):
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                # support a fallback where the file is actually your saved 'states' list
                if isinstance(ckpt, (list, tuple)):
                    # choose first dict-like slot (common format: [state_dict, opt, epoch, step, ...])
                    sd_fallback = None
                    for el in ckpt:
                        if isinstance(el, dict):
                            sd_fallback = el
                            break
                    if sd_fallback is None:
                        sd_fallback = ckpt[0]
                elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    sd_fallback = ckpt['state_dict']
                else:
                    sd_fallback = ckpt

                model_ref = self.model_diff.module if isinstance(self.model_diff, torch.nn.DataParallel) else self.model_diff
                res = model_ref.load_state_dict(sd_fallback, strict=False)
                logging.info(f"Loaded training ckpt (fallback): missing={len(res.missing_keys)} | unexpected={len(res.unexpected_keys)}")
            except Exception as e:
                logging.warning(f"Fallback load failed: {e}")


    def create_pose_model(self, model_path=None):
        config = self.config

        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2, 3]
        edges = torch.tensor(
            [[0,1],[1,2],[2,3],
             [0,4],[4,5],[5,6],
             [0,7],[7,8],[8,9],[9,10],
             [8,11],[11,12],[12,13],
             [8,14],[14,15],[15,16]],
            dtype=torch.long
        )
        adj = adj_mx_from_edges_pose(num_pts=17, edges=edges, sparse=False)

        logging.info("Creating IGCNpose model...")
        logging.info(f"Coords dimensions set to: {config.model.coords_dim}")

        self.model_pose = GCNpose(adj.to(self.device), config).to(self.device)
        self.model_pose = torch.nn.DataParallel(self.model_pose)

        # Params info
        total_params = sum(p.numel() for p in self.model_pose.parameters())
        trainable_params = sum(p.numel() for p in self.model_pose.parameters() if p.requires_grad)
        logging.info(f"IGCNpose model created - Total parameters: {total_params}, Trainable: {trainable_params}")

        if model_path:
            logging.info('Loading pose model from: ' + model_path)
            states = torch.load(model_path, map_location="cpu")
            try:
                self.model_pose.load_state_dict(states[0])
                logging.info("IGCNpose model loaded successfully")
            except Exception as e:
                logging.warning(f"Direct loading failed: {e}")
                logging.info("Attempting to load compatible parameters...")
                model_dict = self.model_pose.state_dict()
                pretrained_dict = states[0] if isinstance(states, (list, tuple)) else states
                compatible_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(compatible_dict)
                self.model_pose.load_state_dict(model_dict)
                logging.info(f"Loaded {len(compatible_dict)}/{len(model_dict)} parameters from checkpoint")
        else:
            logging.info('Initializing pose model with random weights')

    # --------------------------- train ----------------------------

    def train(self):
        logging.info("Starting training...")
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_p1, best_epoch = 1000, 0

        # dataloader
        if config.data.dataset == "human36m":
            logging.info("Creating training data loader...")
            poses_train, poses_train_2d, actions_train, camerapara_train = \
                fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride=1)

            logging.info(f"Training data: {len(poses_train)} sets, {len(poses_train_2d)} 2D sets")

            data_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,
                num_workers=config.training.num_workers, pin_memory=True
            )

            logging.info(f"Training data loader created with {len(data_loader)} batches of size {config.training.batch_size}")
        else:
            raise KeyError('Invalid dataset')

        logging.info("Setting up optimizer...")
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        logging.info(f"Optimizer: {type(optimizer).__name__} with lr={config.optim.lr}")

        if self.config.model.ema:
            logging.info("Initializing EMA...")
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
            logging.info(f"EMA initialized with rate {self.config.model.ema_rate}")
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        # ------------------ Optional resume (robust to list/dict checkpoint formats) ------------------
        if self.args.resume is not None and os.path.isfile(self.args.resume):
            logging.info(f"Resuming from {self.args.resume}")
            ckpt = torch.load(self.args.resume, map_location='cpu')

            # ckpt may be a list (your saved 'states' list) or a dict wrapper.
            # Normalize to a dict-like object for keys/fields.
            resume_state = None
            if isinstance(ckpt, dict):
                # if they saved {'state_dict': ..., 'optimizer': ..., 'epoch': ..., 'step': ...}
                resume_state = ckpt
            elif isinstance(ckpt, (list, tuple)):
                # your saved 'states' is likely [state_dict, optimizer_state, epoch, step, maybe ema]
                # try to decode commonly used positions
                # prefer if the first element is a dict (state_dict)
                if len(ckpt) > 0 and isinstance(ckpt[0], dict):
                    resume_state = {}
                    resume_state['state_dict'] = ckpt[0]
                    # try to extract optimizer if present
                    if len(ckpt) > 1 and isinstance(ckpt[1], dict):
                        resume_state['optimizer'] = ckpt[1]
                    # epoch/step possibly at indices 2/3
                    if len(ckpt) > 2 and isinstance(ckpt[2], int):
                        resume_state['epoch'] = ckpt[2]
                    if len(ckpt) > 3 and isinstance(ckpt[3], int):
                        resume_state['step'] = ckpt[3]
                else:
                    # fallback: try to find dict element inside list
                    for el in ckpt:
                        if isinstance(el, dict):
                            resume_state = {'state_dict': el}
                            break
            else:
                resume_state = None

            # 0) network weights load if present
            if resume_state is not None and 'state_dict' in resume_state and isinstance(resume_state['state_dict'], dict):
                try:
                    # prefer strict load if exact
                    self.model_diff.module.load_state_dict(resume_state['state_dict'], strict=True)
                    logging.info("✔ resume weights loaded (strict).")
                except Exception:
                    logging.warning("Resume ckpt has no exact 'state_dict' match; attempting relaxed load.")
                    try:
                        self.model_diff.module.load_state_dict(resume_state['state_dict'], strict=False)
                        logging.info("✔ resume weights loaded (relaxed).")
                    except Exception as e:
                        logging.warning(f"Relaxed load also failed: {e}. Continuing without loading weights.")

            else:
                logging.warning("Resume ckpt has no 'state_dict'; skipping weights load.")

            # 1) optimiser
            if resume_state is not None and 'optimizer' in resume_state and isinstance(resume_state['optimizer'], dict):
                try:
                    optimizer.load_state_dict(resume_state['optimizer'])
                    logging.info("✔ optimizer state restored from resume ckpt.")
                except Exception as e:
                    logging.warning(f"Could not restore optimizer state: {e}")

            # 2–3) counters
            start_epoch = -1
            step = 0
            if resume_state is not None:
                if 'epoch' in resume_state and isinstance(resume_state['epoch'], int):
                    start_epoch = resume_state['epoch']
                if 'step' in resume_state and isinstance(resume_state['step'], int):
                    step = resume_state['step']
            # finalize
            start_epoch = start_epoch + 1 if start_epoch >= 0 else 0

            logging.info(f"✔ checkpoint parsed — continuing at epoch {start_epoch}")

        
        
        else:
            logging.info("No resume checkpoint given; training from scratch")
        # ----------------------------------------------------------------------------------------------


        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
        logging.info(f"Initial lr={lr_init}, decay={decay}, gamma={gamma}")

        for epoch in range(start_epoch, self.config.training.n_epochs):
            logging.info(f"Starting epoch {epoch}")
            data_start = time.time()
            data_time = 0

            torch.set_grad_enabled(True)
            self.model_diff.train()

            self._set_deq_iterations()
            # --- one-liner banner for current iterations (optional) ---
            if getattr(self.config, 'deq', None) and getattr(self.config.deq, 'enabled', False):
                iters = getattr(self.config.deq, 'iterations', getattr(self.config.deq, 'default_iterations', 15))
                logging.info(f"DEQ: using {iters} iterations (train)")

            epoch_loss_diff = AverageMeter()

            for i, (targets_uvxyz, targets_noise_scale, _, targets_3d, _, _) in enumerate(data_loader):
                data_time += time.time() - data_start
                step += 1

                # Debug first batch
                """
                if i == 0:
                    logging.info("First batch shapes:")
                    logging.info(f"  targets_uvxyz: {targets_uvxyz.shape}")
                    logging.info(f"  targets_noise_scale: {targets_noise_scale.shape}")
                    logging.info(f"  targets_3d: {targets_3d.shape}")
                    logging.info(f"  targets_uvxyz min/max/mean: {targets_uvxyz.min().item():.4f}/{targets_uvxyz.max().item():.4f}/{targets_uvxyz.mean().item():.4f}")
                    logging.info(f"  targets_noise_scale min/max/mean: {targets_noise_scale.min().item():.4f}/{targets_noise_scale.max().item():.4f}/{targets_noise_scale.mean().item():.4f}")
                    logging.info(f"  targets_3d min/max/mean: {targets_3d.min().item():.4f}/{targets_3d.max().item():.4f}/{targets_3d.mean().item():.4f}")
                """

                # to device
                targets_uvxyz = targets_uvxyz.to(self.device)
                targets_noise_scale = targets_noise_scale.to(self.device)
                targets_3d = targets_3d.to(self.device)

                # DDIM/denoising training objective
                n = targets_3d.size(0)
                x = targets_uvxyz
                e = torch.randn_like(x)
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,), device=self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                e = e * (targets_noise_scale)
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()

                """
                if i == 0:
                    logging.info("Diffusion parameters:")
                    logging.info(f"  t range: {t.min().item()}-{t.max().item()}")
                    logging.info(f"  a min/max/mean: {a.min().item():.4f}/{a.max().item():.4f}/{a.mean().item():.4f}")
                    logging.info(f"  Noise e min/max/mean: {e.min().item():.4f}/{e.max().item():.4f}/{e.mean().item():.4f}")
                    logging.info(f"  Noised x min/max/mean: {x.min().item():.4f}/{x.max().item():.4f}/{x.mean().item():.4f}")
                """
                output_noise = self.model_diff(x, self.src_mask, t.float(), 0)
                """
                if i == 0:
                    logging.info("Model outputs:")
                    logging.info(f"  output_noise shape: {output_noise.shape}")
                    logging.info(f"  output_noise min/max/mean: {output_noise.min().item():.4f}/{output_noise.max().item():.4f}/{output_noise.mean().item():.4f}")
                """
                loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)

                if i == 0:
                    logging.info(f"  Initial loss: {loss_diff.item():.6f}")

                optimizer.zero_grad()
                loss_diff.backward()
                torch.nn.utils.clip_grad_norm_(self.model_diff.parameters(), config.optim.grad_clip)
                optimizer.step()

                epoch_loss_diff.update(loss_diff.item(), n)

                if self.config.model.ema:
                    ema_helper.update(self.model_diff)

                if i % 100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'
                                 .format(epoch, i + 1, len(data_loader), step, data_time, epoch_loss_diff.avg))

            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma)
                logging.info(f"Learning rate decayed to {lr_now}")

            # checkpoint & eval every epoch
            logging.info(f"Saving checkpoint at epoch {epoch}")
            states = [
                self.model_diff.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())

            os.makedirs(self.args.log_path, exist_ok=True)
            torch.save(states, os.path.join(self.args.log_path, f"ckpt_{epoch}.pth"))
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            logging.info(f"Saved checkpoint at epoch {epoch}")

            logging.info('test the performance of current model')
            p1, p2 = self.test_hyber(is_train=True)

            if p1 < best_p1:
                best_p1 = p1
                best_epoch = epoch
                torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                logging.info(f"Saved best checkpoint with MPJPE: {p1:.2f}")

            logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'
                         .format(best_epoch, best_p1, epoch, p1, p2))

    # ---------------------------- eval -----------------------------

    def test_hyber(self, is_train=False, is_best_epoch=False):
        """
        Test function that correctly applies DEQ configuration for best epochs
        """
        cudnn.benchmark = True
        args, config = self.args, self.config
        self._set_deq_iterations()

        # --- one-liner banner for current iterations (optional) ---
        if getattr(self.config, 'deq', None) and getattr(self.config.deq, 'enabled', False):
            iters = getattr(self.config.deq, 'iterations', getattr(self.config.deq, 'default_iterations', 15))
            logging.info(f"DEQ: using {iters} iterations (eval)")

        # fixed test parameters
        test_times = args.test_times
        test_timesteps = args.test_timesteps
        test_num_diffusion_timesteps = args.test_num_diffusion_timesteps
        logging.info(f"Using fixed test parameters: times={test_times}, steps={test_timesteps}, diffusion_timesteps={test_num_diffusion_timesteps}")

        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride=1)

            logging.info(f"Validation data: {len(poses_valid)} samples")

            data_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.training.num_workers, pin_memory=True
            )
            logging.info(f"Validation data loader created with {len(data_loader)} batches")
        else:
            raise KeyError('Invalid dataset')

        data_start = time.time()
        data_time = 0

        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()

        # diffusion step schedule
        if args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = list(range(0, test_num_diffusion_timesteps, skip))
        elif args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        max_seq_value = self.num_timesteps - 1
        seq = [min(s, max_seq_value) for s in seq]
        logging.info(f"Diffusion steps: {len(seq)}, sequence: {seq}")

        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',
                                 'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        action_error_sum = define_error_list(self.test_action_list)

        for i, (_, input_noise_scale, input_2d, targets_3d, input_action, camera_para) in enumerate(data_loader):
            data_time += time.time() - data_start

            input_noise_scale = input_noise_scale.to(self.device)
            input_2d = input_2d.to(self.device)
            targets_3d = targets_3d.to(self.device)

            # Pose backbone to get xyz from 2d
            inputs_xyz = self.model_pose(input_2d, self.src_mask)

            if i == 0:
                logging.info(f"First batch GCNpose output: shape={inputs_xyz.shape}, "
                             f"min={inputs_xyz.min().item():.4f}, max={inputs_xyz.max().item():.4f}, "
                             f"mean={inputs_xyz.mean().item():.4f}")

            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :]
            input_uvxyz = torch.cat([input_2d, inputs_xyz], dim=2)

            # repeat for sampling
            input_uvxyz = input_uvxyz.repeat(test_times, 1, 1)
            input_noise_scale = input_noise_scale.repeat(test_times, 1, 1)

            # select final diffusion step
            t = torch.ones(input_uvxyz.size(0), dtype=torch.long, device=self.device) * min(
                test_num_diffusion_timesteps, self.num_timesteps - 1
            )

            # prepare diffusion inputs
            x = input_uvxyz.clone()
            e = torch.randn_like(input_uvxyz)
            b = self.betas
            e = e * input_noise_scale
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)

            try:
                output_uvxyz = generalized_steps(x, self.src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)

                # measure memory & time around generalized_steps
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_time = t1 - t0
                batch_peak_bytes = torch.cuda.max_memory_allocated()
                batch_peak_gb = batch_peak_bytes / (1024**3)

                # optionally log LAST_DEQ_STATS if tracking is enabled
                if getattr(self.config.deq, 'track_residuals', False):
                    last_iters = getattr(LAST_DEQ_STATS, "used_iters_f", None)
                    last_res = (LAST_DEQ_STATS.f_residuals[-1] if getattr(LAST_DEQ_STATS, "f_residuals", None) else float('nan'))
                else:
                    last_iters = None
                    last_res = None
                    
                # run the sampler / generalized steps
                raw_out = generalized_steps(x, self.src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
                logging.debug(f"generalized_steps returned type={type(raw_out)}; repr_len={len(str(raw_out)) if not torch.is_tensor(raw_out) else raw_out.shape}")

                # Robust normalization to a tensor of shape (N_out, 17, 5),
                # where N_out == test_times * batch (if test_times>1) or batch (if test_times==1).
                def _normalize_generalized_output(raw):
                    # case: tuple/list wrappers (common)
                    if isinstance(raw, (tuple, list)):
                        # If tuple like (all_steps_list, other_info), try to find the useful tensor
                        first = raw[0]
                        # often first is a list of stepwise tensors -> take the last step
                        if isinstance(first, (list, tuple)) and len(first) > 0:
                            candidate = first[-1]
                        else:
                            candidate = first
                    else:
                        candidate = raw

                    # candidate should now be a torch.Tensor or array-like
                    if not torch.is_tensor(candidate):
                        try:
                            candidate = torch.as_tensor(candidate, device=self.device)
                        except Exception:
                            raise RuntimeError("generalized_steps returned an unexpected non-tensor object")

                    # Now handle shapes:
                    # If candidate is flat length 5 -> likely single sample -> reshape to (1,17,5)
                    if candidate.dim() == 1 and candidate.numel() == 17 * 5:
                        candidate = candidate.view(1, 17, 5)
                    elif candidate.dim() == 1 and candidate.numel() == 5:
                        # extreme odd case: return single joint vector
                        candidate = candidate.view(1, 1, 5)
                    elif candidate.dim() == 2:
                        # candidate shape maybe (B, 85) -> reshape to (B,17,5) if possible
                        if candidate.size(1) == 17 * 5:
                            candidate = candidate.view(candidate.size(0), 17, 5)
                        else:
                            # if it's (85,) or (batch,) this will raise to make bug visible
                            raise RuntimeError(f"Unexpected 2D candidate shape {tuple(candidate.shape)}")
                    elif candidate.dim() == 3:
                        # expected (N, 17, 5) — good
                        pass
                    else:
                        raise RuntimeError(f"Unexpected candidate tensor dims {candidate.dim()} for generalized_steps output")

                    return candidate

                try:
                    output_uvxyz_all = _normalize_generalized_output(raw_out)  # shape (N_out, 17, 5)
                except Exception as e:
                    logging.error(f"[normalize_output] failed: {e}; raw_out type={type(raw_out)}; raw_out=<{str(raw_out)[:200]}>")
                    raise

                # Now average across test_times if needed.
                # We expect N_out == test_times * batch_size (because earlier we did input_uvxyz.repeat(test_times,1,1))
                N_out = output_uvxyz_all.size(0)
                batch_est = N_out // max(1, test_times)

                if test_times > 1:
                    # ensure divisibility
                    if N_out % test_times != 0:
                        logging.warning(f"Output length {N_out} not divisible by test_times={test_times}; attempting fallback reshape")
                        # fallback: if N_out == batch (test_times=1) we continue, else we raise
                        if N_out == batch_est:
                            output_uvxyz = output_uvxyz_all
                        else:
                            raise RuntimeError(f"Cannot reshape outputs: N_out={N_out} test_times={test_times}")
                    else:
                        output_uvxyz = output_uvxyz_all.view(test_times, -1, 17, 5).mean(0)
                else:
                    # test_times == 1 -> just use tensor as batch x 17 x 5
                    output_uvxyz = output_uvxyz_all

                output_xyz = output_uvxyz[:, :, 2:]


                output_xyz[:, :, :] -= output_xyz[:, :1, :]
                targets_3d[:, :, :] -= targets_3d[:, :1, :]

                # ---------- safe scale (prevents NaN cascade into SVD) ----------
                # Example: global scale using mean absolute magnitudes
                num = targets_3d.abs().mean()
                den = output_xyz.abs().mean()
                scale_factor = _safe_scale(num, den, default=1.0)
                # If tensor scalar, make it a python float for logging
                sf_val = float(scale_factor.detach().cpu()) if scale_factor.numel() == 1 else float(scale_factor.mean().detach().cpu())
                output_xyz = output_xyz * scale_factor
                if i == 0:
                    logging.info(f"Applied scale factor: {sf_val:.4f}")
                # -----------------------------------------------------------------

                current_mpjpe = mpjpe(output_xyz, targets_3d).item() * 1000.0
                current_p_mpjpe = p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0

                if i == 0:
                    logging.info(f"First batch MPJPE: {current_mpjpe:.4f}, P-MPJPE: {current_p_mpjpe:.4f}")

                epoch_loss_3d_pos.update(current_mpjpe, targets_3d.size(0))
                epoch_loss_3d_pos_procrustes.update(current_p_mpjpe, targets_3d.size(0))

                action_error_sum = test_calculation(output_xyz, targets_3d, input_action, action_error_sum, None, None)

            except Exception as e:
                logging.error(f"Error in generalized_steps: {e}")
                raise

            data_start = time.time()

            if i % 100 == 0:
                logging.info('({batch}/{size}) Data: {data:.3f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'
                             .format(batch=i + 1, size=len(data_loader), data=data_time,
                                     e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg))

        logging.info('Final results | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'
                     .format(e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg))

        p1, p2 = print_error(None, action_error_sum, is_train)

        logging.info(f"Standard evaluation with {self.deq_schedule['iterations']} iterations completed")
        logging.info(f"Final MPJPE: {p1:.4f}, P-MPJPE: {p2:.4f}")

        return p1, p2

    def run_deq_sweep(self, iters_list, out_csv="deq_sweep_results.csv", warmup_clear_cuda=True):
        """
        Run full evaluation (test_hyber) for multiple DEQ iteration budgets.

        Parameters
        ----------
        iters_list : list[int]
            list of integer iteration budgets to evaluate (applied via self.deq_schedule['iterations'])
        out_csv : str
            output CSV path to save results (mpjpe, p-mpjpe, wall_seconds, peak_gpu_gb)
        warmup_clear_cuda : bool
            whether to reset CUDA memory stats between runs (recommended)
        """
        import csv, time, torch, statistics
        from models.deq_wrapper import LAST_DEQ_STATS

        results = []
        original_iters = self.deq_schedule.get('iterations', None)

        logging.info(f"Starting DEQ sweep for iterations: {iters_list}; writing to {out_csv}")
        for it in iters_list:
            logging.info(f"--- DEQ sweep run: iterations={it} ---")
            # apply and log change
            self.deq_schedule['iterations'] = it
            self._set_deq_iterations()

            # reset CUDA peak stats and optionally GC
            if warmup_clear_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            try:
                p1, p2 = self.test_hyber(is_train=False)
            except Exception as e:
                logging.error(f"Exception during test_hyber for iterations={it}: {e}")
                # if test fails, add sentinel row and continue
                results.append((it, float("nan"), float("nan"), float("nan"), float("nan"), f"ERROR:{e}"))
                continue
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0

            # peak memory (bytes) for this run
            peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_gb = peak_bytes / (1024**3)

            # optional: gather per-run DEQ stats if available
            used_iters = getattr(LAST_DEQ_STATS, "used_iters_f", None)
            final_res = (LAST_DEQ_STATS.f_residuals[-1] if getattr(LAST_DEQ_STATS, "f_residuals", None) else None)

            logging.info(f"[SWEEP RESULT] it={it} mpjpe={p1:.4f} p-mpjpe={p2:.4f} time_s={elapsed:.2f} peak_GB={peak_gb:.3f} used_iters={used_iters} final_res={final_res}")

            results.append((it, p1, p2, elapsed, peak_gb, used_iters))

        # restore original iterations
        if original_iters is not None:
            self.deq_schedule['iterations'] = original_iters
            self._set_deq_iterations()

        # write CSV
        try:
            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iterations", "mpjpe", "p_mpjpe", "time_s", "peak_gpu_gb", "used_iters"])
                for row in results:
                    writer.writerow(row)
            logging.info(f"Saved sweep CSV to {out_csv}")
        except Exception as e:
            logging.warning(f"Failed to write CSV {out_csv}: {e}")

        return results
