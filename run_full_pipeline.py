"""
Full Pipeline: Learn-to-Trust + C&S
====================================

Stage 1: Learn-to-Trust (3 rounds)
  - Iteratively learn high-quality trusted nodes
  - Test both threshold=0.8 and threshold=0.9

Stage 2: C&S with Expanded Sources
  - Use training nodes + trusted nodes as correction sources
  - Compare with original C&S (training only)

Experiments:
  - Base: MLP predictions (no post-processing)
  - C&S (Original): C&S with training nodes only
  - C&S (Trust-0.8): C&S with training + trusted (threshold=0.8)
  - C&S (Trust-0.9): C&S with training + trusted (threshold=0.9)

Parameters:
  - Learn-to-Trust: alpha=0.999, iterations=50 (for learning)
  - C&S: alpha=0.999, iterations=200 (for final prediction)

Usage:
    python run_full_pipeline.py
    python run_full_pipeline.py --rounds 5

Output:
    - Comparison table (Island/Bridge accuracy)
    - Full_Pipeline_Results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from config import CITIES, NUM_CLASSES, device
from data_utils import load_city_data, get_train_test_masks
from trainer import get_base_predictions
from evaluation import compute_metrics


# =============================================================================
# Helper Functions
# =============================================================================

def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    """Compute prediction margin (top1 - top2 probability)."""
    sorted_p = torch.sort(probs, dim=1, descending=True)[0]
    return sorted_p[:, 0] - sorted_p[:, 1]


def compute_hop_distance(edge_index: torch.Tensor, 
                         source_mask: torch.Tensor,
                         num_nodes: int,
                         max_hops: int = 50) -> torch.Tensor:
    """Compute hop distance from each node to nearest source node using BFS."""
    edge_index_cpu = edge_index.cpu()
    row, col = edge_index_cpu[0].numpy(), edge_index_cpu[1].numpy()
    
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in zip(row, col):
        adj_list[i].append(j)
    
    distances = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    
    source_indices = torch.where(source_mask)[0].cpu().numpy()
    queue = deque()
    
    for idx in source_indices:
        distances[idx] = 0
        queue.append((idx, 0))
    
    visited = set(source_indices)
    
    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return distances


def compute_symmetric_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute symmetric normalization D^{-0.5} A D^{-0.5}."""
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm_weights


def propagate_labels_symmetric(source_signal: torch.Tensor,
                                edge_index: torch.Tensor,
                                norm_weights: torch.Tensor,
                                num_nodes: int,
                                alpha: float = 0.999,
                                num_iter: int = 50) -> torch.Tensor:
    """C&S-style label propagation with symmetric normalization."""
    row, col = edge_index
    num_classes = source_signal.size(1)
    H = source_signal
    Z = H.clone()
    for _ in range(num_iter):
        Z_new = torch.zeros_like(Z)
        Z_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                          Z[col] * norm_weights.unsqueeze(1))
        Z = (1 - alpha) * H + alpha * Z_new
    return Z


# =============================================================================
# Learn-to-Trust Module
# =============================================================================

class LearnToTrustModule(nn.Module):
    """Learnable confidence weights for injection nodes."""
    
    def __init__(self, num_nodes: int, learnable_mask: torch.Tensor, 
                 frozen_mask: torch.Tensor, init_value: float = 2.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.learnable_mask = learnable_mask
        self.frozen_mask = frozen_mask
        
        num_learnable = learnable_mask.sum().item()
        if num_learnable > 0:
            self.raw_weights = nn.Parameter(torch.full((num_learnable,), init_value, device=device))
        else:
            self.raw_weights = None
        
        self.learnable_indices = torch.where(learnable_mask)[0]
        self.frozen_indices = torch.where(frozen_mask)[0]
    
    def get_confidence_weights(self) -> torch.Tensor:
        weights = torch.zeros(self.num_nodes, device=device)
        if len(self.frozen_indices) > 0:
            weights[self.frozen_indices] = 1.0
        if self.raw_weights is not None and len(self.learnable_indices) > 0:
            weights[self.learnable_indices] = torch.sigmoid(self.raw_weights)
        return weights
    
    def forward(self, pred_probs, margins, edge_index, norm_weights, alpha=0.999, num_iter=50):
        confidence_weights = self.get_confidence_weights()
        preds = pred_probs.argmax(dim=1)
        pred_onehot = F.one_hot(preds, num_classes=pred_probs.size(1)).float()
        injection_mask = self.learnable_mask | self.frozen_mask
        weights = confidence_weights * margins
        source_signal = pred_onehot * weights.unsqueeze(1)
        source_signal = source_signal * injection_mask.unsqueeze(1).float()
        propagated = propagate_labels_symmetric(source_signal, edge_index, norm_weights, self.num_nodes, alpha, num_iter)
        return propagated


def train_learn_to_trust(model, pred_probs, margins, edge_index, norm_weights,
                         pseudo_labels, validation_mask, alpha=0.999, 
                         num_prop_iter=50, num_epochs=200, lr=0.01, verbose=False):
    """Train confidence weights."""
    if model.raw_weights is None:
        return model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    valid_nodes = validation_mask & (pseudo_labels >= 0)
    val_labels = pseudo_labels[valid_nodes].long()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        propagated = model(pred_probs, margins, edge_index, norm_weights, alpha, num_prop_iter)
        propagated_at_val = propagated[valid_nodes]
        propagated_probs = propagated_at_val / (propagated_at_val.sum(dim=1, keepdim=True) + 1e-10)
        log_probs = torch.log(propagated_probs + 1e-10)
        loss = F.nll_loss(log_probs, val_labels)
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"        Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return model


# =============================================================================
# Learn-to-Trust Stage
# =============================================================================

def run_learn_to_trust(base_probs, y, edge_index, train_mask, bridge_mask, 
                       num_nodes, num_rounds=3, trust_threshold=0.9,
                       alpha=0.999, prop_iter=50, max_hops=50,
                       epochs=200, lr=0.01, verbose=False):
    """
    Run Learn-to-Trust for multiple rounds.
    
    Returns:
        trusted_mask: Boolean mask of trusted nodes
        pseudo_labels: Labels for all nodes (real for training, pseudo for trusted)
    """
    island_mask = ~bridge_mask
    margins = compute_margin(base_probs)
    preds = base_probs.argmax(dim=1)
    norm_weights = compute_symmetric_norm(edge_index, num_nodes)
    
    # Separate training nodes
    island_train_mask = train_mask & island_mask
    bridge_train_mask = train_mask & bridge_mask
    
    # Initialize
    pseudo_labels = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    pseudo_labels[train_mask] = y[train_mask].long()
    
    island_anchor_mask = island_train_mask.clone()
    frozen_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    all_trusted_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    
    for round_num in range(num_rounds):
        if verbose:
            print(f"      Round {round_num}...")
        
        # Compute reachable nodes
        hop_dist_from_island = compute_hop_distance(edge_index, island_anchor_mask, num_nodes, max_hops)
        within_island_range = (hop_dist_from_island >= 0) & (hop_dist_from_island <= max_hops)
        
        hop_dist_from_bridge = compute_hop_distance(edge_index, bridge_train_mask, num_nodes, max_hops)
        within_bridge_range = (hop_dist_from_bridge >= 0) & (hop_dist_from_bridge <= max_hops)
        
        # Dual validation zone
        dual_validation_zone = within_island_range & within_bridge_range & ~train_mask & ~all_trusted_mask
        dual_validation_active = dual_validation_zone.sum().item() > 0
        
        # Nodes to validate
        validate_mask = within_island_range & ~train_mask & ~all_trusted_mask
        new_reachable = validate_mask.sum().item()
        
        if new_reachable == 0:
            break
        
        # Validation nodes for loss
        validation_nodes = island_anchor_mask.clone()
        if dual_validation_active:
            validation_nodes = validation_nodes | bridge_train_mask
        
        # Create and train model
        learnable_mask = validate_mask
        model = LearnToTrustModule(num_nodes, learnable_mask, frozen_mask).to(device)
        model = train_learn_to_trust(model, base_probs, margins, edge_index, norm_weights,
                                     pseudo_labels, validation_nodes, alpha, prop_iter,
                                     epochs, lr, verbose=verbose)
        
        # Get trusted nodes
        with torch.no_grad():
            learned_weights = model.get_confidence_weights()
        newly_trusted = learnable_mask & (learned_weights > trust_threshold)
        
        new_trusted_count = newly_trusted.sum().item()
        if verbose:
            print(f"        New trusted: {new_trusted_count}")
        
        if new_trusted_count == 0:
            break
        
        # Update
        pseudo_labels[newly_trusted] = preds[newly_trusted]
        all_trusted_mask = all_trusted_mask | newly_trusted
        frozen_mask = all_trusted_mask.clone()
        trusted_island = all_trusted_mask & island_mask
        island_anchor_mask = island_train_mask | trusted_island
    
    return all_trusted_mask, pseudo_labels


# =============================================================================
# C&S Implementation
# =============================================================================

def run_correct_and_smooth(base_probs, edge_index, labels, source_mask, num_nodes, num_classes,
                           alpha_correct=0.999, num_iter_correct=200,
                           alpha_smooth=0.999, num_iter_smooth=200):
    """
    Run Correct & Smooth.
    
    Args:
        base_probs: Base model predictions (N, C)
        edge_index: Graph edges
        labels: Ground truth / pseudo labels for source nodes
        source_mask: Mask of correction source nodes
        num_nodes: Number of nodes
        num_classes: Number of classes
        alpha_correct: Propagation alpha for correct phase (default: 0.999)
        num_iter_correct: Iterations for correct phase (default: 200)
        alpha_smooth: Propagation alpha for smooth phase (default: 0.999)
        num_iter_smooth: Iterations for smooth phase (default: 200)
    
    Returns:
        Final predictions (N, C)
    """
    probs = base_probs.clone()
    row, col = edge_index
    
    # Compute normalization
    deg = torch.zeros(num_nodes, device=probs.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=probs.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    valid_source = source_mask & (labels >= 0)
    source_labels = F.one_hot(labels[valid_source].long(), num_classes).float()
    
    # === CORRECT PHASE ===
    E = torch.zeros(num_nodes, num_classes, device=probs.device)
    E[valid_source] = source_labels - probs[valid_source]
    
    E_prop = E.clone()
    for _ in range(num_iter_correct):
        E_new = torch.zeros_like(E_prop)
        E_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                          E_prop[col] * norm_weights.unsqueeze(1))
        E_prop = (1 - alpha_correct) * E + alpha_correct * E_new
    
    probs = probs + E_prop
    probs = probs.clamp(min=0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
    probs[valid_source] = source_labels
    
    # === SMOOTH PHASE ===
    Z = probs.clone()
    for _ in range(num_iter_smooth):
        Z_new = torch.zeros_like(Z)
        Z_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                          Z[col] * norm_weights.unsqueeze(1))
        Z = (1 - alpha_smooth) * probs + alpha_smooth * Z_new
    
    Z = Z / (Z.sum(dim=1, keepdim=True) + 1e-10)
    Z[valid_source] = source_labels
    
    return Z


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
    parser = argparse.ArgumentParser(description='Full Pipeline: Learn-to-Trust + C&S')
    parser.add_argument('--ratio', type=str, default='05pct',
                        choices=['05pct', '10pct', '15pct', '20pct', '25pct'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=3, help='Learn-to-Trust rounds')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs per round')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--l2t_alpha', type=float, default=0.999, help='L2T propagation alpha')
    parser.add_argument('--l2t_iter', type=int, default=50, help='L2T propagation iterations')
    parser.add_argument('--cs_alpha', type=float, default=0.999, help='C&S propagation alpha')
    parser.add_argument('--cs_iter', type=int, default=200, help='C&S propagation iterations')
    parser.add_argument('--max_hops', type=int, default=50, help='Max hops per round')
    parser.add_argument('--output', type=str, default='Full_Pipeline_Results.csv')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL PIPELINE: Learn-to-Trust + C&S")
    print("="*70)
    print(f"Base Model: MLP")
    print(f"Cities: {CITIES}")
    print(f"Ratio: {args.ratio}")
    print(f"Learn-to-Trust: {args.rounds} rounds, alpha={args.l2t_alpha}, iter={args.l2t_iter}")
    print(f"Trust thresholds: [0.8, 0.9]")
    print(f"C&S: alpha={args.cs_alpha}, iter={args.cs_iter}")
    print(f"Device: {device}")
    print("="*70)
    
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
        
        # Train base model
        print("\n  Training base model (MLP)...")
        base_probs = get_base_predictions(
            model_name='MLP',
            in_dim=data_dict['in_dim'],
            x=x, y=y,
            train_mask=train_mask,
            edge_index=edge_index,
            seed=args.seed
        )
        
        # === Method 1: Base Model ===
        print("\n  [1] Base Model (no post-processing)...")
        base_results = evaluate_predictions(base_probs, y, test_mask, bridge_mask)
        
        # === Method 2: C&S (Original) ===
        print("  [2] C&S (Original - training nodes only)...")
        cs_original_probs = run_correct_and_smooth(
            base_probs, edge_index, y, train_mask, num_nodes, NUM_CLASSES,
            alpha_correct=args.cs_alpha, num_iter_correct=args.cs_iter,
            alpha_smooth=args.cs_alpha, num_iter_smooth=args.cs_iter
        )
        cs_original_results = evaluate_predictions(cs_original_probs, y, test_mask, bridge_mask)
        
        # === Method 3: C&S (Trust-0.8) ===
        print("  [3] Learn-to-Trust (threshold=0.8) + C&S...")
        trusted_mask_08, pseudo_labels_08 = run_learn_to_trust(
            base_probs, y, edge_index, train_mask, bridge_mask, num_nodes,
            num_rounds=args.rounds, trust_threshold=0.8,
            alpha=args.l2t_alpha, prop_iter=args.l2t_iter, max_hops=args.max_hops,
            epochs=args.epochs, lr=args.lr, verbose=args.verbose
        )
        
        # Combine sources: training + trusted
        source_mask_08 = train_mask | trusted_mask_08
        trusted_count_08 = trusted_mask_08.sum().item()
        print(f"      Trusted nodes: {trusted_count_08}")
        
        cs_trust08_probs = run_correct_and_smooth(
            base_probs, edge_index, pseudo_labels_08, source_mask_08, num_nodes, NUM_CLASSES,
            alpha_correct=args.cs_alpha, num_iter_correct=args.cs_iter,
            alpha_smooth=args.cs_alpha, num_iter_smooth=args.cs_iter
        )
        cs_trust08_results = evaluate_predictions(cs_trust08_probs, y, test_mask, bridge_mask)
        
        # === Method 4: C&S (Trust-0.9) ===
        print("  [4] Learn-to-Trust (threshold=0.9) + C&S...")
        trusted_mask_09, pseudo_labels_09 = run_learn_to_trust(
            base_probs, y, edge_index, train_mask, bridge_mask, num_nodes,
            num_rounds=args.rounds, trust_threshold=0.9,
            alpha=args.l2t_alpha, prop_iter=args.l2t_iter, max_hops=args.max_hops,
            epochs=args.epochs, lr=args.lr, verbose=args.verbose
        )
        
        source_mask_09 = train_mask | trusted_mask_09
        trusted_count_09 = trusted_mask_09.sum().item()
        print(f"      Trusted nodes: {trusted_count_09}")
        
        cs_trust09_probs = run_correct_and_smooth(
            base_probs, edge_index, pseudo_labels_09, source_mask_09, num_nodes, NUM_CLASSES,
            alpha_correct=args.cs_alpha, num_iter_correct=args.cs_iter,
            alpha_smooth=args.cs_alpha, num_iter_smooth=args.cs_iter
        )
        cs_trust09_results = evaluate_predictions(cs_trust09_probs, y, test_mask, bridge_mask)
        
        # === Print Comparison ===
        print(f"\n  {'='*60}")
        print(f"  RESULTS: {city}")
        print(f"  {'='*60}")
        
        methods = [
            ('Base', base_results, 0),
            ('C&S (Original)', cs_original_results, train_mask.sum().item()),
            ('C&S (Trust-0.8)', cs_trust08_results, source_mask_08.sum().item()),
            ('C&S (Trust-0.9)', cs_trust09_results, source_mask_09.sum().item()),
        ]
        
        print(f"\n  {'Method':<20} {'Sources':>10} {'Island F1':>12} {'Bridge F1':>12}")
        print(f"  {'-'*56}")
        
        for method_name, results, source_count in methods:
            island_f1 = results['Island']['Macro_F1'] * 100
            bridge_f1 = results['Bridge']['Macro_F1'] * 100
            print(f"  {method_name:<20} {source_count:>10} {island_f1:>11.2f}% {bridge_f1:>11.2f}%")
            
            # Store results
            for region in ['Island', 'Bridge']:
                all_results.append({
                    'City': city,
                    'Method': method_name,
                    'Sources': source_count,
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
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY: Macro-F1 by Method")
    print("="*70)
    
    summary_df = df.pivot_table(
        values='Macro_F1', 
        index=['City', 'Region'], 
        columns='Method'
    ).reset_index()
    
    print("\nIsland Macro-F1:")
    print(f"  {'City':<12} {'Base':>10} {'C&S Orig':>12} {'Trust-0.8':>12} {'Trust-0.9':>12}")
    print(f"  {'-'*60}")
    for city in CITIES:
        city_island = df[(df['City'] == city) & (df['Region'] == 'Island')]
        if len(city_island) == 0:
            continue
        base = city_island[city_island['Method'] == 'Base']['Macro_F1'].values[0] * 100
        cs_orig = city_island[city_island['Method'] == 'C&S (Original)']['Macro_F1'].values[0] * 100
        cs_08 = city_island[city_island['Method'] == 'C&S (Trust-0.8)']['Macro_F1'].values[0] * 100
        cs_09 = city_island[city_island['Method'] == 'C&S (Trust-0.9)']['Macro_F1'].values[0] * 100
        print(f"  {city:<12} {base:>10.2f}% {cs_orig:>11.2f}% {cs_08:>11.2f}% {cs_09:>11.2f}%")
    
    print("\nBridge Macro-F1:")
    print(f"  {'City':<12} {'Base':>10} {'C&S Orig':>12} {'Trust-0.8':>12} {'Trust-0.9':>12}")
    print(f"  {'-'*60}")
    for city in CITIES:
        city_bridge = df[(df['City'] == city) & (df['Region'] == 'Bridge')]
        if len(city_bridge) == 0:
            continue
        base = city_bridge[city_bridge['Method'] == 'Base']['Macro_F1'].values[0] * 100
        cs_orig = city_bridge[city_bridge['Method'] == 'C&S (Original)']['Macro_F1'].values[0] * 100
        cs_08 = city_bridge[city_bridge['Method'] == 'C&S (Trust-0.8)']['Macro_F1'].values[0] * 100
        cs_09 = city_bridge[city_bridge['Method'] == 'C&S (Trust-0.9)']['Macro_F1'].values[0] * 100
        print(f"  {city:<12} {base:>10.2f}% {cs_orig:>11.2f}% {cs_08:>11.2f}% {cs_09:>11.2f}%")


if __name__ == "__main__":
    main()
