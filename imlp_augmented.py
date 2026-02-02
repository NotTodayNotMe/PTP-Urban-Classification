"""
Modified IM-LP with Feature + Label Augmentation
=================================================

Key Modification:
  - MLP receives [features | propagated_labels] as input
  - At each stage (1+), re-propagate labels and re-predict

Flow:
  Initialization:
    - LP from training nodes → propagated_labels
    - Train MLP on [features | propagated_labels]
    - MLP prediction → probs

  Stage 0:
    - Correct(probs) → corrected_probs
    - Smooth(corrected_probs) → probs

  Stage 1+:
    - RE-PROPAGATE: LP from ALL nodes using current probs → new propagated_labels
    - RE-PREDICT: MLP([features | new propagated_labels]) → probs
    - Correct(probs) → corrected_probs
    - Smooth(corrected_probs) → probs

Usage:
    python imlp_augmented.py
    python imlp_augmented.py --num_stages 5

Output:
    - Comparison table (Island/Bridge metrics)
    - IMLP_Augmented_Results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from config import CITIES, NUM_CLASSES, device, EPOCHS, PATIENCE, HIDDEN_DIM
from data_utils import load_city_data, get_train_test_masks
from evaluation import compute_metrics


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IMLPAugmentedConfig:
    """Configuration for Augmented IM-LP."""
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
    # LP for feature augmentation
    lp_alpha: float = 0.999
    lp_iter: int = 200


# =============================================================================
# MLP Model
# =============================================================================

class MLP(nn.Module):
    """MLP that accepts [features | propagated_labels] as input."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        return x


# =============================================================================
# Helper Functions
# =============================================================================

def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    """Compute prediction margin (top1 - top2 probability)."""
    sorted_p = torch.sort(probs, dim=1, descending=True)[0]
    return sorted_p[:, 0] - sorted_p[:, 1]


def label_propagation(source_labels: torch.Tensor,
                      edge_index: torch.Tensor,
                      num_nodes: int,
                      alpha: float = 0.999,
                      num_iter: int = 200) -> torch.Tensor:
    """
    Simple label propagation.
    
    Args:
        source_labels: Label distribution for all nodes (N, C)
                       Can be one-hot for training nodes, probs for others
        edge_index: Graph edges (2, E)
        num_nodes: Number of nodes
        alpha: Propagation weight
        num_iter: Number of iterations
    
    Returns:
        Propagated labels (N, C)
    """
    row, col = edge_index
    num_classes = source_labels.size(1)
    
    # Compute degree normalization (symmetric)
    deg = torch.zeros(num_nodes, device=source_labels.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=source_labels.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    # Propagate
    H = source_labels
    Z = H.clone()
    
    for _ in range(num_iter):
        Z_new = torch.zeros_like(Z)
        Z_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                          Z[col] * norm_weights.unsqueeze(1))
        Z = (1 - alpha) * H + alpha * Z_new
    
    # Normalize to probabilities
    Z = Z / (Z.sum(dim=1, keepdim=True) + 1e-10)
    
    return Z


def train_mlp(model: MLP,
              x: torch.Tensor,
              y: torch.Tensor,
              train_mask: torch.Tensor,
              epochs: int = EPOCHS,
              patience: int = PATIENCE,
              lr: float = 0.01,
              weight_decay: float = 5e-4) -> MLP:
    """Train MLP model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    valid_train = train_mask & (y >= 0)
    
    best_model_state = None
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(x)
        loss = F.cross_entropy(logits[valid_train], y[valid_train])
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    return model


# =============================================================================
# Augmented IM-LP
# =============================================================================

class IMLPAugmented:
    """
    IM-LP with Feature + Label Augmentation.
    
    Key difference from original:
      - MLP receives [features | propagated_labels] as input
      - At each stage (1+), re-propagate and re-predict before correct/smooth
    """
    
    def __init__(self, config: IMLPAugmentedConfig = None):
        self.config = config or IMLPAugmentedConfig()
        self.mlp = None
    
    def __call__(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 edge_index: torch.Tensor,
                 train_mask: torch.Tensor,
                 num_nodes: int,
                 num_classes: int) -> torch.Tensor:
        """
        Run Augmented IM-LP.
        
        Args:
            x: Node features (N, F)
            y: Node labels (N,)
            edge_index: Graph edges (2, E)
            train_mask: Training node mask (N,)
            num_nodes: Number of nodes
            num_classes: Number of classes
        
        Returns:
            Final predictions (N, C)
        """
        cfg = self.config
        
        train_mask = train_mask.bool() if not train_mask.dtype == torch.bool else train_mask
        valid_train = train_mask & (y >= 0)
        y_long = y.long()
        
        # Precompute graph normalization
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv = 1.0 / deg.clamp(min=1)
        
        # === INITIALIZATION ===
        # Initial LP from training nodes (one-hot)
        initial_labels = torch.zeros(num_nodes, num_classes, device=x.device)
        initial_labels[valid_train] = F.one_hot(y_long[valid_train], num_classes).float()
        
        propagated_labels = label_propagation(
            initial_labels, edge_index, num_nodes, 
            alpha=cfg.lp_alpha, num_iter=cfg.lp_iter
        )
        
        # Augment features with propagated labels
        x_augmented = torch.cat([x, propagated_labels], dim=1)
        
        # Train MLP on augmented features
        in_dim = x_augmented.size(1)
        self.mlp = MLP(in_dim, HIDDEN_DIM, num_classes).to(device)
        self.mlp = train_mlp(self.mlp, x_augmented, y_long, train_mask)
        
        # Initial prediction
        self.mlp.eval()
        with torch.no_grad():
            logits = self.mlp(x_augmented)
            probs = F.softmax(logits, dim=1)
        
        # Tracking for stability
        stable_rounds = torch.zeros(num_nodes, device=x.device)
        prev_preds = probs.argmax(dim=1)
        current_gamma = cfg.gamma
        
        # === STAGES ===
        for stage in range(cfg.num_stages):
            
            # Stage 1+: Re-propagate and re-predict
            if stage > 0:
                # Re-propagate using current probs from ALL nodes
                propagated_labels = label_propagation(
                    probs, edge_index, num_nodes,
                    alpha=cfg.lp_alpha, num_iter=cfg.lp_iter
                )
                
                # Re-augment features
                x_augmented = torch.cat([x, propagated_labels], dim=1)
                
                # Re-predict using trained MLP
                with torch.no_grad():
                    logits = self.mlp(x_augmented)
                    probs = F.softmax(logits, dim=1)
            
            # === CORRECT STAGE ===
            probs = self._correct_stage(
                probs, y_long, valid_train, row, col, deg_inv,
                num_nodes, num_classes, current_gamma
            )
            current_gamma *= cfg.gamma_decay
            
            # === SMOOTH STAGE ===
            probs, stable_rounds, prev_preds = self._smooth_stage(
                probs, y_long, valid_train, row, col, deg_inv,
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


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_predictions(probs, y, test_mask, bridge_mask):
    """Evaluate predictions on Island and Bridge test nodes."""
    island_test_mask = test_mask & ~bridge_mask
    bridge_test_mask = test_mask & bridge_mask
    
    island_metrics = compute_metrics(probs, y, island_test_mask)
    bridge_metrics = compute_metrics(probs, y, bridge_test_mask)
    
    return {
        'Island': island_metrics,
        'Bridge': bridge_metrics
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Augmented IM-LP')
    parser.add_argument('--ratio', type=str, default='05pct',
                        choices=['05pct', '10pct', '15pct', '20pct', '25pct'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_stages', type=int, default=3, help='Number of stages')
    parser.add_argument('--gamma', type=float, default=0.5, help='Correction strength')
    parser.add_argument('--conf_threshold', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='IMLP_Augmented_Results.csv')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*70)
    print("AUGMENTED IM-LP: Feature + Label")
    print("="*70)
    print(f"Cities: {CITIES}")
    print(f"Ratio: {args.ratio}")
    print(f"Stages: {args.num_stages}")
    print(f"Gamma: {args.gamma}")
    print(f"Conf threshold: {args.conf_threshold}")
    print(f"Device: {device}")
    print("="*70)
    
    config = IMLPAugmentedConfig(
        num_stages=args.num_stages,
        gamma=args.gamma,
        conf_threshold=args.conf_threshold
    )
    
    all_results = []
    
    for city in CITIES:
        print(f"\n{'#'*70}")
        print(f"Processing {city}...")
        print(f"{'#'*70}")
        
        # Load data
        try:
            data_dict = load_city_data(city, args.ratio)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue
        
        x = data_dict['x']
        y = data_dict['y']
        edge_index = data_dict['edge_index']
        train_mask = data_dict['train_mask']
        test_mask = data_dict['test_mask']
        bridge_mask = data_dict['bridge_mask']
        num_nodes = data_dict['num_nodes']
        
        print(f"  Nodes: {num_nodes}, Train: {train_mask.sum().item()}")
        
        # Run Augmented IM-LP
        print(f"\n  Running Augmented IM-LP...")
        imlp = IMLPAugmented(config)
        probs = imlp(x, y, edge_index, train_mask, num_nodes, NUM_CLASSES)
        
        # Evaluate
        results = evaluate_predictions(probs, y, test_mask, bridge_mask)
        
        # Print results
        print(f"\n  Results:")
        print(f"    Island - Macro F1: {results['Island']['Macro_F1']*100:.2f}%")
        print(f"    Bridge - Macro F1: {results['Bridge']['Macro_F1']*100:.2f}%")
        
        # Store results
        for region in ['Island', 'Bridge']:
            all_results.append({
                'City': city,
                'Region': region,
                'AUC': results[region]['AUC'],
                'Recall': results[region]['Recall'],
                'Macro_F1': results[region]['Macro_F1'],
                'Micro_F1': results[region]['Micro_F1']
            })
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Macro-F1")
    print("="*70)
    print(f"\n{'City':<12} {'Island F1':>12} {'Bridge F1':>12}")
    print(f"{'-'*38}")
    for city in CITIES:
        city_df = df[df['City'] == city]
        if len(city_df) == 0:
            continue
        island_f1 = city_df[city_df['Region'] == 'Island']['Macro_F1'].values[0] * 100
        bridge_f1 = city_df[city_df['Region'] == 'Bridge']['Macro_F1'].values[0] * 100
        print(f"{city:<12} {island_f1:>11.2f}% {bridge_f1:>11.2f}%")


if __name__ == "__main__":
    main()
