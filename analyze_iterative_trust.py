"""
Iterative Learn-to-Trust Expansion (V2)
=======================================

Updates from V1:
1. Validate using both real labels AND pseudo-labels from trusted nodes
2. Dual validation for Bridge: activate Bridge training only when nodes 
   are reachable by BOTH Bridge training AND trusted Island nodes
3. Stricter threshold: weight > 0.9 (instead of 0.8)

Algorithm:
  Round 0: 
    - Anchors = Island training (real labels)
    - Validate within 50 hops
    - Store pseudo-labels for trusted nodes
    
  Round 1+:
    - Anchors = Island training + trusted nodes (with pseudo-labels)
    - If dual validation zone exists (reachable by both Island anchors AND Bridge training):
      - Activate Bridge training for validation
    - Validate, store pseudo-labels, freeze

Usage:
    python analyze_iterative_trust.py
    python analyze_iterative_trust.py --rounds 5 --trust_threshold 0.95

Output:
    - Console tables showing per-round statistics
    - Iterative_Trust_V2.csv with all results
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
    """
    Compute hop distance from each node to nearest source node using BFS.
    
    Returns:
        Distance tensor (N,) with -1 for unreachable nodes
    """
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
# Iterative Learn to Trust Module (V2)
# =============================================================================

class IterativeLearnToTrustV2(nn.Module):
    """
    Learnable confidence weights with support for frozen nodes.
    
    - Frozen nodes: Fixed weight = 1 (from previous rounds)
    - Learnable nodes: Sigmoid-constrained weights
    """
    
    def __init__(self, num_nodes: int, learnable_mask: torch.Tensor, 
                 frozen_mask: torch.Tensor, init_value: float = 2.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.learnable_mask = learnable_mask
        self.frozen_mask = frozen_mask
        
        # Learnable weights for non-frozen injection nodes
        num_learnable = learnable_mask.sum().item()
        if num_learnable > 0:
            self.raw_weights = nn.Parameter(torch.full((num_learnable,), init_value, device=device))
        else:
            self.raw_weights = None
        
        # Mapping
        self.learnable_indices = torch.where(learnable_mask)[0]
        self.frozen_indices = torch.where(frozen_mask)[0]
    
    def get_confidence_weights(self) -> torch.Tensor:
        """Get confidence weights for all nodes."""
        weights = torch.zeros(self.num_nodes, device=device)
        
        # Frozen nodes get weight = 1
        if len(self.frozen_indices) > 0:
            weights[self.frozen_indices] = 1.0
        
        # Learnable nodes get sigmoid(raw_weight)
        if self.raw_weights is not None and len(self.learnable_indices) > 0:
            weights[self.learnable_indices] = torch.sigmoid(self.raw_weights)
        
        return weights
    
    def forward(self, 
                pred_probs: torch.Tensor,
                margins: torch.Tensor,
                edge_index: torch.Tensor,
                norm_weights: torch.Tensor,
                alpha: float = 0.999,
                num_iter: int = 50) -> torch.Tensor:
        """Propagate weighted labels from injection nodes."""
        confidence_weights = self.get_confidence_weights()
        
        preds = pred_probs.argmax(dim=1)
        pred_onehot = F.one_hot(preds, num_classes=pred_probs.size(1)).float()
        
        # Weight by confidence and margin
        injection_mask = self.learnable_mask | self.frozen_mask
        weights = confidence_weights * margins
        source_signal = pred_onehot * weights.unsqueeze(1)
        source_signal = source_signal * injection_mask.unsqueeze(1).float()
        
        propagated = propagate_labels_symmetric(
            source_signal, edge_index, norm_weights, self.num_nodes, alpha, num_iter
        )
        
        return propagated


def train_iterative_trust_v2(model: IterativeLearnToTrustV2,
                              pred_probs: torch.Tensor,
                              margins: torch.Tensor,
                              edge_index: torch.Tensor,
                              norm_weights: torch.Tensor,
                              pseudo_labels: torch.Tensor,
                              validation_mask: torch.Tensor,
                              alpha: float = 0.999,
                              num_prop_iter: int = 50,
                              num_epochs: int = 200,
                              lr: float = 0.01) -> IterativeLearnToTrustV2:
    """
    Train confidence weights using validation nodes (training + trusted with pseudo-labels).
    
    Args:
        model: IterativeLearnToTrustV2 module
        pred_probs: Base model predictions (N, C)
        margins: Prediction margins (N,)
        edge_index: Graph edges
        norm_weights: Symmetric normalization weights
        pseudo_labels: Labels for validation (real for training, pseudo for trusted)
        validation_mask: Mask of nodes to use for validation
        ...
    
    Returns:
        Trained model
    """
    if model.raw_weights is None:
        print("      No learnable weights. Skipping training.")
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
        
        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return model


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_round_results(trusted_mask: torch.Tensor,
                          pred_probs: torch.Tensor,
                          y: torch.Tensor,
                          test_mask: torch.Tensor,
                          island_mask: torch.Tensor,
                          bridge_mask: torch.Tensor) -> dict:
    """Analyze newly trusted nodes in this round."""
    preds = pred_probs.argmax(dim=1)
    correct = (preds == y)
    
    results = {}
    
    for region_name, region_mask in [('Island', island_mask), ('Bridge', bridge_mask)]:
        # Newly trusted nodes in this region that are also test nodes
        mask = trusted_mask & region_mask & test_mask
        
        count = mask.sum().item()
        if count > 0:
            accuracy = correct[mask].float().mean().item() * 100
        else:
            accuracy = 0.0
        
        results[region_name] = {
            'count': count,
            'accuracy': accuracy
        }
    
    return results


def print_round_summary(round_num: int, results: dict, 
                        total_trusted: int, coverage_pct: float,
                        new_reachable: int, dual_validation: bool) -> None:
    """Print summary for a single round."""
    print(f"\n  --- Round {round_num} Results ---")
    print(f"  New nodes within 50 hops: {new_reachable}")
    print(f"  Coverage: {coverage_pct:.2f}%")
    print(f"  Dual validation (Bridge training active): {'YES' if dual_validation else 'NO'}")
    print(f"\n  Newly Trusted Nodes:")
    print(f"    {'Region':<10} {'Count':>10} {'Accuracy':>12}")
    print(f"    {'-'*34}")
    
    total_new = 0
    for region in ['Island', 'Bridge']:
        r = results[region]
        print(f"    {region:<10} {r['count']:>10} {r['accuracy']:>11.2f}%")
        total_new += r['count']
    
    print(f"    {'-'*34}")
    print(f"    {'Total New':<10} {total_new:>10}")
    print(f"    {'Cumulative':<10} {total_trusted:>10}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Iterative Learn-to-Trust Expansion V2')
    parser.add_argument('--ratio', type=str, default='05pct',
                        choices=['05pct', '10pct', '15pct', '20pct', '25pct'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=3, help='Number of expansion rounds')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs per round')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.999, help='Propagation weight')
    parser.add_argument('--prop_iter', type=int, default=50, help='Propagation iterations')
    parser.add_argument('--max_hops', type=int, default=50, help='Max hops per round')
    parser.add_argument('--trust_threshold', type=float, default=0.9, help='Threshold for trusted (stricter)')
    parser.add_argument('--init_weight', type=float, default=2.0, help='Initial raw weight')
    parser.add_argument('--output', type=str, default='Iterative_Trust_V2.csv')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ITERATIVE LEARN-TO-TRUST EXPANSION (V2)")
    print("="*70)
    print(f"Base Model: MLP")
    print(f"Cities: {CITIES}")
    print(f"Ratio: {args.ratio}")
    print(f"Rounds: {args.rounds}")
    print(f"Epochs per round: {args.epochs}, LR: {args.lr}")
    print(f"Propagation: alpha={args.alpha}, iterations={args.prop_iter}")
    print(f"Trust threshold: weight > {args.trust_threshold} (STRICTER)")
    print(f"Max hops per round: {args.max_hops}")
    print(f"Initial anchors: Training nodes (ISLAND ONLY)")
    print(f"Validation: Real labels + Pseudo-labels from trusted nodes")
    print(f"Dual validation: Activate Bridge training when reachable by both sides")
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
        
        island_mask = ~bridge_mask
        
        # Separate Island and Bridge training nodes
        island_train_mask = train_mask & island_mask
        bridge_train_mask = train_mask & bridge_mask
        
        island_train_count = island_train_mask.sum().item()
        bridge_train_count = bridge_train_mask.sum().item()
        
        print(f"\n  Training nodes breakdown:")
        print(f"    Island training: {island_train_count}")
        print(f"    Bridge training: {bridge_train_count} (for dual validation)")
        
        # Get base model predictions
        print("\n  Training base model (MLP)...")
        base_probs = get_base_predictions(
            model_name='MLP',
            in_dim=data_dict['in_dim'],
            x=x, y=y,
            train_mask=train_mask,
            edge_index=edge_index,
            seed=args.seed
        )
        margins = compute_margin(base_probs)
        preds = base_probs.argmax(dim=1)
        
        # Compute symmetric normalization
        norm_weights = compute_symmetric_norm(edge_index, num_nodes)
        
        # Initialize tracking
        # Pseudo-labels: start with real labels for training nodes, -1 for others
        pseudo_labels = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        pseudo_labels[train_mask] = y[train_mask].long()
        
        # Anchor mask: start with Island training only
        island_anchor_mask = island_train_mask.clone()
        
        # Frozen mask: nodes with fixed weight = 1
        frozen_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        # All trusted mask: accumulates trusted nodes across rounds
        all_trusted_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        cumulative_trusted = 0
        city_results = []
        
        for round_num in range(args.rounds):
            print(f"\n  ====== Round {round_num} ======")
            
            # Compute reachable nodes from current Island anchors
            hop_dist_from_island = compute_hop_distance(edge_index, island_anchor_mask, num_nodes, args.max_hops)
            within_island_range = (hop_dist_from_island >= 0) & (hop_dist_from_island <= args.max_hops)
            
            # Compute reachable nodes from Bridge training
            hop_dist_from_bridge = compute_hop_distance(edge_index, bridge_train_mask, num_nodes, args.max_hops)
            within_bridge_range = (hop_dist_from_bridge >= 0) & (hop_dist_from_bridge <= args.max_hops)
            
            # Dual validation zone: reachable by BOTH Island anchors AND Bridge training
            dual_validation_zone = within_island_range & within_bridge_range & ~train_mask & ~all_trusted_mask
            dual_validation_active = dual_validation_zone.sum().item() > 0
            
            # Nodes to validate: within Island range, not training, not already trusted
            validate_mask = within_island_range & ~train_mask & ~all_trusted_mask
            
            new_reachable = validate_mask.sum().item()
            coverage_pct = within_island_range.sum().item() / num_nodes * 100
            
            print(f"  Island anchors: {island_anchor_mask.sum().item()}")
            print(f"  New nodes to validate: {new_reachable}")
            print(f"  Dual validation zone: {dual_validation_zone.sum().item()} nodes")
            
            if new_reachable == 0:
                print("  No new nodes to validate. Stopping.")
                break
            
            # Determine validation mask for loss computation
            # Always use Island anchors (training + trusted)
            validation_nodes = island_anchor_mask.clone()
            
            # If dual validation active, also add Bridge training nodes
            if dual_validation_active:
                validation_nodes = validation_nodes | bridge_train_mask
                print(f"  Dual validation ACTIVE: using Island anchors + Bridge training for loss")
            else:
                print(f"  Dual validation NOT active: using Island anchors only for loss")
            
            # Learnable nodes: within range and not already trusted
            learnable_mask = validate_mask
            
            # Create and train model
            model = IterativeLearnToTrustV2(
                num_nodes=num_nodes,
                learnable_mask=learnable_mask,
                frozen_mask=frozen_mask,
                init_value=args.init_weight
            ).to(device)
            
            print("  Training confidence weights...")
            model = train_iterative_trust_v2(
                model=model,
                pred_probs=base_probs,
                margins=margins,
                edge_index=edge_index,
                norm_weights=norm_weights,
                pseudo_labels=pseudo_labels,
                validation_mask=validation_nodes,
                alpha=args.alpha,
                num_prop_iter=args.prop_iter,
                num_epochs=args.epochs,
                lr=args.lr
            )
            
            # Get learned weights
            with torch.no_grad():
                learned_weights = model.get_confidence_weights()
            
            # Identify newly trusted nodes: high weight among learnable nodes
            newly_trusted = learnable_mask & (learned_weights > args.trust_threshold)
            
            # Analyze results
            round_results = analyze_round_results(
                trusted_mask=newly_trusted,
                pred_probs=base_probs,
                y=y,
                test_mask=test_mask,
                island_mask=island_mask,
                bridge_mask=bridge_mask
            )
            
            # Update tracking
            new_trusted_count = newly_trusted.sum().item()
            cumulative_trusted += new_trusted_count
            
            # Print round summary
            print_round_summary(round_num, round_results, cumulative_trusted, 
                              coverage_pct, new_reachable, dual_validation_active)
            
            # Store results
            for region in ['Island', 'Bridge']:
                city_results.append({
                    'City': city,
                    'Round': round_num,
                    'Region': region,
                    'New_Reachable': new_reachable,
                    'Coverage_Pct': coverage_pct,
                    'Dual_Validation': dual_validation_active,
                    'Newly_Trusted': round_results[region]['count'],
                    'Accuracy': round_results[region]['accuracy'],
                    'Cumulative_Trusted': cumulative_trusted
                })
            
            # Update for next round
            # Store pseudo-labels for newly trusted nodes
            pseudo_labels[newly_trusted] = preds[newly_trusted]
            
            # Update masks
            all_trusted_mask = all_trusted_mask | newly_trusted
            frozen_mask = all_trusted_mask.clone()  # Freeze all trusted so far
            
            # Update Island anchors: Island training + trusted Island nodes
            trusted_island = all_trusted_mask & island_mask
            island_anchor_mask = island_train_mask | trusted_island
            
            if new_trusted_count == 0:
                print("  No new trusted nodes. Stopping early.")
                break
        
        # Final summary for city
        print(f"\n  {'='*50}")
        print(f"  FINAL SUMMARY: {city}")
        print(f"  {'='*50}")
        print(f"  Total trusted nodes: {cumulative_trusted}")
        
        # Breakdown by region
        final_island = (all_trusted_mask & island_mask & test_mask).sum().item()
        final_bridge = (all_trusted_mask & bridge_mask & test_mask).sum().item()
        
        correct = (preds == y)
        
        island_acc = correct[all_trusted_mask & island_mask & test_mask].float().mean().item() * 100 if final_island > 0 else 0
        bridge_acc = correct[all_trusted_mask & bridge_mask & test_mask].float().mean().item() * 100 if final_bridge > 0 else 0
        
        print(f"\n  Final Trusted Breakdown (Test Nodes):")
        print(f"    Island: {final_island} nodes, {island_acc:.2f}% accuracy")
        print(f"    Bridge: {final_bridge} nodes, {bridge_acc:.2f}% accuracy")
        
        final_coverage = (island_train_mask | all_trusted_mask).sum().item()
        print(f"\n  Final anchor coverage: {final_coverage} nodes ({100*final_coverage/num_nodes:.2f}%)")
        
        all_results.extend(city_results)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}")
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY: Cumulative Trusted by Round")
    print("="*70)
    
    for city in CITIES:
        city_df = df[df['City'] == city]
        if len(city_df) == 0:
            continue
        
        print(f"\n{city}:")
        print(f"  {'Round':<8} {'Island':>12} {'Island Acc':>12} {'Bridge':>12} {'Bridge Acc':>12} {'Dual Val':>10}")
        print(f"  {'-'*68}")
        
        for round_num in city_df['Round'].unique():
            round_df = city_df[city_df['Round'] == round_num]
            island_row = round_df[round_df['Region'] == 'Island'].iloc[0]
            bridge_row = round_df[round_df['Region'] == 'Bridge'].iloc[0]
            dual_val = 'YES' if island_row['Dual_Validation'] else 'NO'
            
            print(f"  {round_num:<8} {island_row['Newly_Trusted']:>12} {island_row['Accuracy']:>11.2f}% "
                  f"{bridge_row['Newly_Trusted']:>12} {bridge_row['Accuracy']:>11.2f}% {dual_val:>10}")


if __name__ == "__main__":
    main()
