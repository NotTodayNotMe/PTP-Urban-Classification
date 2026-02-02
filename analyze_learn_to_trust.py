"""
Learn to Trust - Step 1 Verification (Updated)
==============================================

Test whether we can learn confidence weights for pseudo-labels
by validating propagated labels against training node ground truth.

Updates from v1:
- Loss: MSE -> Cross-entropy
- Normalization: D^{-1}A -> D^{-0.5} A D^{-0.5} (symmetric)
- Init: sigmoid(0)=0.5 -> sigmoid(2)≈0.88
- Epochs: 50 -> 200

Usage:
    python analyze_learn_to_trust.py
    python analyze_learn_to_trust.py --epochs 200 --lr 0.01

Output:
    - Console tables showing learned weight vs accuracy
    - Learn_to_Trust_Step1.csv with all results
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
    
    Args:
        edge_index: Graph edges (2, E)
        source_mask: Boolean mask of source nodes
        num_nodes: Total number of nodes
        max_hops: Maximum hops to search
    
    Returns:
        Distance tensor (N,) with -1 for unreachable nodes
    """
    # Build adjacency list on CPU for BFS
    edge_index_cpu = edge_index.cpu()
    row, col = edge_index_cpu[0].numpy(), edge_index_cpu[1].numpy()
    
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in zip(row, col):
        adj_list[i].append(j)
    
    # Initialize distances
    distances = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    
    # BFS from all source nodes
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
    """
    Compute symmetric normalization D^{-0.5} A D^{-0.5}.
    
    Returns:
        norm_weights: Normalization weights for each edge
    """
    row, col = edge_index
    
    # Compute degree
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
    
    # D^{-0.5}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Normalization weights: D^{-0.5}[i] * D^{-0.5}[j] for each edge (i,j)
    norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    return norm_weights


def propagate_labels_symmetric(source_signal: torch.Tensor,
                                edge_index: torch.Tensor,
                                norm_weights: torch.Tensor,
                                num_nodes: int,
                                alpha: float = 0.999,
                                num_iter: int = 50) -> torch.Tensor:
    """
    C&S-style label propagation with symmetric normalization.
    
    Args:
        source_signal: Initial signal at source nodes (N, C)
        edge_index: Graph edges (2, E)
        norm_weights: Precomputed D^{-0.5} A D^{-0.5} weights
        num_nodes: Number of nodes
        alpha: Propagation weight
        num_iter: Number of iterations
    
    Returns:
        Propagated signal (N, C)
    """
    row, col = edge_index
    num_classes = source_signal.size(1)
    
    H = source_signal  # Initial signal
    Z = H.clone()
    
    for _ in range(num_iter):
        # Symmetric normalized aggregation
        Z_new = torch.zeros_like(Z)
        Z_new.scatter_add_(0, row.unsqueeze(1).expand(-1, num_classes),
                          Z[col] * norm_weights.unsqueeze(1))
        Z = (1 - alpha) * H + alpha * Z_new
    
    return Z


# =============================================================================
# Learn to Trust Module
# =============================================================================

class LearnToTrust(nn.Module):
    """
    Learnable confidence weights for injection nodes.
    
    Confidence weights are constrained to [0, 1] using sigmoid.
    Initialized with raw_weight = 2.0 so sigmoid(2) ≈ 0.88.
    """
    
    def __init__(self, num_nodes: int, injection_mask: torch.Tensor, init_value: float = 2.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.injection_mask = injection_mask
        
        # Initialize raw weights to init_value, so sigmoid(init_value) is starting confidence
        num_injection = injection_mask.sum().item()
        self.raw_weights = nn.Parameter(torch.full((num_injection,), init_value, device=device))
        
        # Store mapping from injection index to node index
        self.injection_indices = torch.where(injection_mask)[0]
    
    def get_confidence_weights(self) -> torch.Tensor:
        """Get confidence weights for all nodes (0 for non-injection nodes)."""
        weights = torch.zeros(self.num_nodes, device=device)
        weights[self.injection_indices] = torch.sigmoid(self.raw_weights)
        return weights
    
    def forward(self, 
                pred_probs: torch.Tensor,
                margins: torch.Tensor,
                edge_index: torch.Tensor,
                norm_weights: torch.Tensor,
                alpha: float = 0.999,
                num_iter: int = 50) -> torch.Tensor:
        """
        Propagate weighted labels from injection nodes.
        
        Args:
            pred_probs: Predicted probabilities (N, C)
            margins: Prediction margins (N,)
            edge_index: Graph edges
            norm_weights: Symmetric normalization weights
            alpha: Propagation weight
            num_iter: Number of iterations
        
        Returns:
            Propagated labels at all nodes (N, C)
        """
        confidence_weights = self.get_confidence_weights()
        
        # Create source signal: confidence * margin * one_hot_prediction
        preds = pred_probs.argmax(dim=1)
        pred_onehot = F.one_hot(preds, num_classes=pred_probs.size(1)).float()
        
        # Weight by confidence and margin
        weights = confidence_weights * margins  # (N,)
        source_signal = pred_onehot * weights.unsqueeze(1)  # (N, C)
        
        # Zero out non-injection nodes
        source_signal = source_signal * self.injection_mask.unsqueeze(1).float()
        
        # Propagate with symmetric normalization
        propagated = propagate_labels_symmetric(
            source_signal, edge_index, norm_weights, self.num_nodes, alpha, num_iter
        )
        
        return propagated


def train_confidence_weights(model: LearnToTrust,
                             pred_probs: torch.Tensor,
                             margins: torch.Tensor,
                             edge_index: torch.Tensor,
                             norm_weights: torch.Tensor,
                             y: torch.Tensor,
                             train_mask: torch.Tensor,
                             alpha: float = 0.999,
                             num_prop_iter: int = 50,
                             num_epochs: int = 200,
                             lr: float = 0.01) -> LearnToTrust:
    """
    Train confidence weights by minimizing cross-entropy at training nodes.
    
    Args:
        model: LearnToTrust module
        pred_probs: Base model predictions (N, C)
        margins: Prediction margins (N,)
        edge_index: Graph edges
        norm_weights: Symmetric normalization weights
        y: Ground truth labels (N,)
        train_mask: Training node mask (N,)
        alpha: Propagation weight
        num_prop_iter: Propagation iterations
        num_epochs: Training epochs
        lr: Learning rate
    
    Returns:
        Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    valid_train = train_mask & (y >= 0)
    train_labels = y[valid_train].long()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward: propagate from injection nodes
        propagated = model(pred_probs, margins, edge_index, norm_weights, alpha, num_prop_iter)
        
        # Get propagated signal at training nodes
        propagated_at_train = propagated[valid_train]
        
        # Normalize to get log-probabilities for cross-entropy
        # Add small epsilon to avoid log(0)
        propagated_probs = propagated_at_train / (propagated_at_train.sum(dim=1, keepdim=True) + 1e-10)
        log_probs = torch.log(propagated_probs + 1e-10)
        
        # Cross-entropy loss
        loss = F.nll_loss(log_probs, train_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return model


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_learned_weights(learned_weights: torch.Tensor,
                            pred_probs: torch.Tensor,
                            y: torch.Tensor,
                            analysis_mask: torch.Tensor,
                            island_mask: torch.Tensor,
                            bridge_mask: torch.Tensor) -> dict:
    """
    Analyze learned weights vs accuracy for Island and Bridge nodes.
    
    Args:
        learned_weights: Learned confidence weights (N,)
        pred_probs: Base model predictions (N, C)
        y: Ground truth labels (N,)
        analysis_mask: Mask of nodes to analyze (within 50 hops)
        island_mask: Island node mask
        bridge_mask: Bridge node mask
    
    Returns:
        Dictionary with analysis results
    """
    preds = pred_probs.argmax(dim=1)
    correct = (preds == y)
    
    # Move to CPU
    weights_np = learned_weights.cpu().numpy()
    correct_np = correct.cpu().numpy()
    analysis_mask_np = analysis_mask.cpu().numpy()
    island_mask_np = island_mask.cpu().numpy()
    bridge_mask_np = bridge_mask.cpu().numpy()
    
    # Define weight bins
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    results = {'Island': [], 'Bridge': []}
    
    for region, region_mask_np in [('Island', island_mask_np), ('Bridge', bridge_mask_np)]:
        # Nodes in this region AND within analysis scope AND are injection nodes (weight > 0)
        combined_mask = analysis_mask_np & region_mask_np & (weights_np > 0)
        region_indices = np.where(combined_mask)[0]
        
        region_weights = weights_np[region_indices]
        region_correct = correct_np[region_indices]
        
        total_count = len(region_indices)
        
        for low, high in bins:
            if high == 1.0:
                in_bin = (region_weights >= low) & (region_weights <= high)
            else:
                in_bin = (region_weights >= low) & (region_weights < high)
            
            count = in_bin.sum()
            if count > 0:
                accuracy = region_correct[in_bin].mean() * 100
            else:
                accuracy = 0.0
            
            percentage = (count / total_count * 100) if total_count > 0 else 0.0
            
            results[region].append({
                'Weight_Range': f'[{low:.1f}, {high:.1f}' + (']' if high == 1.0 else ')'),
                'Count': int(count),
                'Accuracy': accuracy,
                'Percentage': percentage
            })
    
    return results


def print_analysis_table(results: dict, city: str, setting: str) -> None:
    """Print formatted analysis table."""
    print(f"\n{'='*60}")
    print(f"LEARNED WEIGHT ANALYSIS: {city} - {setting}")
    print(f"{'='*60}")
    
    for region in ['Island', 'Bridge']:
        print(f"\n{region} Nodes:")
        print(f"{'Weight Range':<15} {'Count':>10} {'Accuracy':>12} {'% of Total':>12}")
        print("-" * 51)
        
        for row in results[region]:
            print(f"{row['Weight_Range']:<15} {row['Count']:>10} {row['Accuracy']:>11.2f}% {row['Percentage']:>11.2f}%")
        
        total_count = sum(r['Count'] for r in results[region])
        if total_count > 0:
            total_correct = sum(r['Count'] * r['Accuracy'] / 100 for r in results[region])
            overall_acc = total_correct / total_count * 100
        else:
            overall_acc = 0.0
        print("-" * 51)
        print(f"{'Total':<15} {total_count:>10} {overall_acc:>11.2f}% {100.0:>11.2f}%")


def results_to_dataframe(all_results: list) -> pd.DataFrame:
    """Convert all results to a DataFrame."""
    rows = []
    for result_entry in all_results:
        city = result_entry['city']
        setting = result_entry['setting']
        results = result_entry['results']
        
        for region in ['Island', 'Bridge']:
            for row in results[region]:
                rows.append({
                    'City': city,
                    'Setting': setting,
                    'Region': region,
                    'Weight_Range': row['Weight_Range'],
                    'Count': row['Count'],
                    'Accuracy': row['Accuracy'],
                    'Percentage': row['Percentage']
                })
    
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Learn to Trust - Step 1 Verification')
    parser.add_argument('--ratio', type=str, default='05pct',
                        choices=['05pct', '10pct', '15pct', '20pct', '25pct'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.999, help='Propagation weight')
    parser.add_argument('--prop_iter', type=int, default=50, help='Propagation iterations')
    parser.add_argument('--max_hops', type=int, default=50, help='Max hops for analysis')
    parser.add_argument('--init_weight', type=float, default=2.0, help='Initial raw weight (sigmoid(x))')
    parser.add_argument('--output', type=str, default='Learn_to_Trust_Step1.csv')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LEARN TO TRUST - STEP 1 VERIFICATION (Updated)")
    print("="*60)
    print(f"Base Model: MLP")
    print(f"Cities: {CITIES}")
    print(f"Ratio: {args.ratio}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Propagation: alpha={args.alpha}, iterations={args.prop_iter}")
    print(f"Normalization: Symmetric (D^-0.5 A D^-0.5)")
    print(f"Loss: Cross-entropy")
    print(f"Init weight: sigmoid({args.init_weight}) = {torch.sigmoid(torch.tensor(args.init_weight)).item():.4f}")
    print(f"Analysis scope: nodes within {args.max_hops} hops of training")
    print(f"Device: {device}")
    print("="*60)
    
    all_results = []
    
    for city in CITIES:
        print(f"\n{'#'*60}")
        print(f"Processing {city}...")
        print(f"{'#'*60}")
        
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
        
        island_test_mask, bridge_test_mask = get_train_test_masks(data_dict)
        
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
        
        # Compute symmetric normalization weights
        print("  Computing symmetric normalization weights...")
        norm_weights = compute_symmetric_norm(edge_index, num_nodes)
        
        # Compute hop distances from training nodes
        print("  Computing hop distances from training nodes...")
        hop_distances = compute_hop_distance(edge_index, train_mask, num_nodes, args.max_hops)
        within_range_mask = (hop_distances >= 0) & (hop_distances <= args.max_hops)
        
        num_within_range = within_range_mask.sum().item()
        print(f"  Nodes within {args.max_hops} hops: {num_within_range} ({100*num_within_range/num_nodes:.2f}%)")
        
        # Test both settings
        settings = [
            ('All_Unlabeled', ~train_mask),
            ('High_Margin_0.8', (~train_mask) & (margins >= 0.8))
        ]
        
        for setting_name, injection_mask in settings:
            num_injection = injection_mask.sum().item()
            print(f"\n  Setting: {setting_name}")
            print(f"  Injection nodes: {num_injection}")
            
            # Create and train model
            model = LearnToTrust(num_nodes, injection_mask, init_value=args.init_weight).to(device)
            
            print("  Training confidence weights...")
            model = train_confidence_weights(
                model=model,
                pred_probs=base_probs,
                margins=margins,
                edge_index=edge_index,
                norm_weights=norm_weights,
                y=y,
                train_mask=train_mask,
                alpha=args.alpha,
                num_prop_iter=args.prop_iter,
                num_epochs=args.epochs,
                lr=args.lr
            )
            
            # Get learned weights
            with torch.no_grad():
                learned_weights = model.get_confidence_weights()
            
            # Analysis mask: test nodes within range
            analysis_mask = test_mask & within_range_mask
            
            # Analyze
            results = analyze_learned_weights(
                learned_weights=learned_weights,
                pred_probs=base_probs,
                y=y,
                analysis_mask=analysis_mask,
                island_mask=~bridge_mask,
                bridge_mask=bridge_mask
            )
            
            # Print table
            print_analysis_table(results, city, setting_name)
            
            # Store results
            all_results.append({
                'city': city,
                'setting': setting_name,
                'results': results
            })
    
    # Save to CSV
    df = results_to_dataframe(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Average Accuracy by Learned Weight Range")
    print("="*60)
    
    for setting in ['All_Unlabeled', 'High_Margin_0.8']:
        print(f"\n--- Setting: {setting} ---")
        setting_df = df[df['Setting'] == setting]
        
        for region in ['Island', 'Bridge']:
            print(f"\n{region}:")
            region_df = setting_df[setting_df['Region'] == region]
            summary = region_df.groupby('Weight_Range').agg({
                'Count': 'sum',
                'Accuracy': 'mean'
            }).reset_index()
            
            weight_order = ['[0.0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']
            summary['sort_key'] = summary['Weight_Range'].apply(lambda x: weight_order.index(x) if x in weight_order else 99)
            summary = summary.sort_values('sort_key').drop('sort_key', axis=1)
            
            total_nodes = summary['Count'].sum()
            
            print(f"{'Weight Range':<15} {'Total Count':>12} {'% of Total':>12} {'Avg Accuracy':>12}")
            print("-" * 53)
            for _, row in summary.iterrows():
                pct = row['Count'] / total_nodes * 100 if total_nodes > 0 else 0
                print(f"{row['Weight_Range']:<15} {row['Count']:>12} {pct:>11.2f}% {row['Accuracy']:>11.2f}%")
            print("-" * 53)
            print(f"{'Total':<15} {total_nodes:>12} {100.0:>11.2f}%")


if __name__ == "__main__":
    main()
