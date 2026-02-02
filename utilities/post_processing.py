"""
Post-processing Methods
=======================
Contains: Label Propagation (LP), Correct & Smooth (C&S), and IM-LP (our method)

This is the main file to modify when experimenting with new methods.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class LPConfig:
    """Configuration for Label Propagation."""
    num_iter: int = 50
    alpha: float = 0.9


@dataclass  
class CSConfig:
    """Configuration for Correct & Smooth."""
    num_iter_correct: int = 200
    num_iter_smooth: int = 200
    alpha_correct: float = 0.999
    alpha_smooth: float = 0.999


@dataclass
class IMLPConfig:
    """
    Configuration for IM-LP.
    
    Attributes:
        num_stages: Number of iterative stages
        gamma: Correction strength
        num_iter_correct: Iterations in correction phase
        num_iter_smooth: Iterations in smoothing phase
        alpha_correct: Propagation weight in correction
        alpha_smooth: Propagation weight in smoothing
        dense_weight: Weight for pseudo-label contributions
        conf_threshold: Confidence threshold for pseudo-labels
        margin_thresh: Margin threshold for soft gating
        stability_thresh: Stability threshold for soft gating
        sharpness: Sharpness of sigmoid gating
        gamma_decay: Decay rate for gamma across stages
    """
    num_stages: int = 3
    gamma: float = 0.5
    gamma_decay: float = 1.0
    num_iter_correct: int = 200
    alpha_correct: float = 0.999
    num_iter_smooth: int = 200
    alpha_smooth: float = 0.999
    dense_weight: float = 0.5
    conf_threshold: float = 0.8
    margin_thresh: float = 0.3
    stability_thresh: float = 0.5
    sharpness: float = 10.0


# =============================================================================
# Helper Functions
# =============================================================================

def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    """Compute prediction margin (top1 - top2 probability)."""
    sorted_p = torch.sort(probs, dim=1, descending=True)[0]
    return sorted_p[:, 0] - sorted_p[:, 1]


# =============================================================================
# Label Propagation (LP)
# =============================================================================

class LabelPropagation:
    """Standard Label Propagation."""
    
    def __init__(self, config: LPConfig = None):
        self.config = config or LPConfig()
    
    def __call__(self, 
                 probs: torch.Tensor,
                 edge_index: torch.Tensor,
                 y: torch.Tensor,
                 train_mask: torch.Tensor,
                 num_nodes: int,
                 num_classes: int) -> torch.Tensor:
        probs = probs.clone()
        train_t = train_mask.bool() if not train_mask.dtype == torch.bool else train_mask
        valid_train = train_t & (y >= 0)
        y_t = y.long()
        
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=probs.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=probs.device))
        deg_inv = 1.0 / deg.clamp(min=1)
        
        Y = probs.clone()
        train_labels = F.one_hot(y_t[valid_train], num_classes).float()
        Y[valid_train] = train_labels
        
        for _ in range(self.config.num_iter):
            Y_new = torch.zeros_like(Y)
            Y_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                              Y[col] * deg_inv[col].unsqueeze(1))
            Y = (1 - self.config.alpha) * probs + self.config.alpha * Y_new
            Y[valid_train] = train_labels
        
        Y = Y / (Y.sum(dim=1, keepdim=True) + 1e-10)
        return Y


# =============================================================================
# Correct & Smooth (C&S)
# =============================================================================

class CorrectAndSmooth:
    """Correct & Smooth (Huang et al., 2021)."""
    
    def __init__(self, config: CSConfig = None):
        self.config = config or CSConfig()
    
    def __call__(self,
                 probs: torch.Tensor,
                 edge_index: torch.Tensor,
                 y: torch.Tensor,
                 train_mask: torch.Tensor,
                 num_nodes: int,
                 num_classes: int) -> torch.Tensor:
        probs = probs.clone()
        train_t = train_mask.bool() if not train_mask.dtype == torch.bool else train_mask
        valid_train = train_t & (y >= 0)
        y_t = y.long()
        
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=probs.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=probs.device))
        deg_inv = 1.0 / deg.clamp(min=1)
        
        train_labels = F.one_hot(y_t[valid_train], num_classes).float()
        
        # CORRECT PHASE
        E = torch.zeros(num_nodes, num_classes, device=probs.device)
        E[valid_train] = train_labels - probs[valid_train]
        
        E_prop = E.clone()
        for _ in range(self.config.num_iter_correct):
            E_new = torch.zeros_like(E_prop)
            E_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                              E_prop[col] * deg_inv[col].unsqueeze(1))
            E_prop = (1 - self.config.alpha_correct) * E + self.config.alpha_correct * E_new
        
        probs = probs + E_prop
        probs = probs.clamp(min=0)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
        probs[valid_train] = train_labels
        
        # SMOOTH PHASE
        Z = probs.clone()
        for _ in range(self.config.num_iter_smooth):
            Z_new = torch.zeros_like(Z)
            Z_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                              Z[col] * deg_inv[col].unsqueeze(1))
            Z = (1 - self.config.alpha_smooth) * probs + self.config.alpha_smooth * Z_new
        
        Z = Z / (Z.sum(dim=1, keepdim=True) + 1e-10)
        Z[valid_train] = train_labels
        
        return Z


# =============================================================================
# IM-LP (Our Method)
# =============================================================================

class IMLP:
    """
    Iterative Margin-weighted Label Propagation (Our Method).
    
    Key Components:
    1. Confident pseudo-label injection: Create stepping stones
    2. Trust-based error correction: Learn which pseudo-labels to trust
    3. Margin-weighted smoothing: Down-weight uncertain boundary nodes
    """
    
    def __init__(self, config: IMLPConfig = None):
        self.config = config or IMLPConfig()
    
    def __call__(self,
                 probs: torch.Tensor,
                 edge_index: torch.Tensor,
                 y: torch.Tensor,
                 train_mask: torch.Tensor,
                 num_nodes: int,
                 num_classes: int) -> torch.Tensor:
        cfg = self.config
        probs = probs.clone()
        
        train_t = train_mask.bool() if not train_mask.dtype == torch.bool else train_mask
        valid_train = train_t & (y >= 0)
        y_t = y.long()
        
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=probs.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=probs.device))
        deg_inv = 1.0 / deg.clamp(min=1)
        
        stable_rounds = torch.zeros(num_nodes, device=probs.device)
        prev_preds = probs.argmax(dim=1)
        current_gamma = cfg.gamma
        
        for stage in range(cfg.num_stages):
            # === CORRECT STAGE ===
            probs = self._correct_stage(
                probs, y_t, valid_train, row, col, deg_inv, 
                num_nodes, num_classes, current_gamma
            )
            current_gamma *= cfg.gamma_decay
            
            # === SMOOTH STAGE ===
            probs, stable_rounds, prev_preds = self._smooth_stage(
                probs, y_t, valid_train, row, col, deg_inv,
                num_nodes, num_classes, stable_rounds, prev_preds, stage
            )
        
        return probs
    
    def _correct_stage(self, probs, y_t, valid_train, row, col, deg_inv,
                       num_nodes, num_classes, gamma):
        """Error correction phase."""
        cfg = self.config
        
        E = torch.zeros(num_nodes, num_classes, device=probs.device)
        
        # Ground truth errors from training nodes
        train_labels = F.one_hot(y_t[valid_train], num_classes).float()
        E[valid_train] = train_labels - probs[valid_train]
        
        # Pseudo-label errors from confident nodes
        margins = compute_margin(probs)
        conf = probs.max(dim=1)[0]
        preds = probs.argmax(dim=1)
        
        confident = (conf > cfg.conf_threshold) & ~valid_train
        if confident.sum() > 0:
            pseudo_labels = F.one_hot(preds[confident], num_classes).float()
            pseudo_errors = pseudo_labels - probs[confident]
            E[confident] = pseudo_errors * margins[confident].unsqueeze(1) * cfg.dense_weight
        
        # Propagate errors
        E_prop = E.clone()
        for _ in range(cfg.num_iter_correct):
            E_new = torch.zeros_like(E_prop)
            E_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                              E_prop[col] * deg_inv[col].unsqueeze(1))
            E_prop = (1 - cfg.alpha_correct) * E + cfg.alpha_correct * E_new
        
        # Apply correction
        probs = probs + gamma * E_prop
        probs = probs.clamp(min=0)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
        probs[valid_train] = F.one_hot(y_t[valid_train], num_classes).float()
        
        return probs
    
    def _smooth_stage(self, probs, y_t, valid_train, row, col, deg_inv,
                      num_nodes, num_classes, stable_rounds, prev_preds, stage):
        """Margin-weighted smoothing phase."""
        cfg = self.config
        
        # Update stability tracking
        curr_preds = probs.argmax(dim=1)
        stable = (curr_preds == prev_preds) | valid_train
        stable_rounds = torch.where(stable, stable_rounds + 1, torch.zeros_like(stable_rounds))
        prev_preds = curr_preds.clone()
        
        stability = stable_rounds / (stage + 1)
        margins = compute_margin(probs)
        
        # Soft threshold gating
        margin_gate = torch.sigmoid(cfg.sharpness * (margins - cfg.margin_thresh))
        stability_gate = torch.sigmoid(cfg.sharpness * (stability - cfg.stability_thresh))
        
        # Compute weights
        weights = torch.zeros(num_nodes, device=probs.device)
        weights[valid_train] = 1.0
        unlabeled = ~valid_train
        weights[unlabeled] = (margin_gate[unlabeled] * stability_gate[unlabeled] * 
                             margins[unlabeled] * stability[unlabeled])
        
        # Weighted smoothing
        H = probs * weights.unsqueeze(1)
        Z = H.clone()
        
        for _ in range(cfg.num_iter_smooth):
            Z_new = torch.zeros_like(Z)
            Z_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                              Z[col] * deg_inv[col].unsqueeze(1))
            Z = (1 - cfg.alpha_smooth) * H + cfg.alpha_smooth * Z_new
        
        probs = Z / (Z.sum(dim=1, keepdim=True) + 1e-10)
        probs[valid_train] = F.one_hot(y_t[valid_train], num_classes).float()
        
        return probs, stable_rounds, prev_preds
