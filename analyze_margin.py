"""
Margin Analysis
===============

Test whether margin can effectively estimate pseudo-label quality.

Goal: Validate if higher margin correlates with higher accuracy
      for both Island and Bridge nodes separately.

Usage:
    python analyze_margin.py
    python analyze_margin.py --ratio 10pct

Output:
    - Console tables showing margin vs accuracy
    - Margin_Analysis_1.csv with all results
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from config import CITIES, NUM_CLASSES, device
from data_utils import load_city_data, get_train_test_masks
from trainer import get_base_predictions


# =============================================================================
# Margin Analysis Functions
# =============================================================================

def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    """Compute prediction margin (top1 - top2 probability)."""
    sorted_p = torch.sort(probs, dim=1, descending=True)[0]
    return sorted_p[:, 0] - sorted_p[:, 1]


def compute_margin_analysis(data_dict: dict, base_probs: torch.Tensor) -> dict:
    """
    Analyze margin vs accuracy for Island and Bridge test nodes.
    
    Args:
        data_dict: Data dictionary from load_city_data
        base_probs: Base model predictions (N, C)
    
    Returns:
        Dictionary with analysis results for Island and Bridge
    """
    y = data_dict['y']
    island_test_mask, bridge_test_mask = get_train_test_masks(data_dict)
    
    # Get predictions and margins
    preds = base_probs.argmax(dim=1)
    margins = compute_margin(base_probs)
    
    # Move to CPU for analysis
    y_np = y.cpu().numpy()
    preds_np = preds.cpu().numpy()
    margins_np = margins.cpu().numpy()
    island_mask_np = island_test_mask.cpu().numpy()
    bridge_mask_np = bridge_test_mask.cpu().numpy()
    
    # Define margin bins
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    results = {'Island': [], 'Bridge': []}
    
    for region, mask_np in [('Island', island_mask_np), ('Bridge', bridge_mask_np)]:
        # Get data for this region
        region_indices = np.where(mask_np)[0]
        region_y = y_np[region_indices]
        region_preds = preds_np[region_indices]
        region_margins = margins_np[region_indices]
        region_correct = (region_y == region_preds)
        
        total_count = len(region_indices)
        
        for low, high in bins:
            # Find nodes in this margin range
            if high == 1.0:
                in_bin = (region_margins >= low) & (region_margins <= high)
            else:
                in_bin = (region_margins >= low) & (region_margins < high)
            
            count = in_bin.sum()
            if count > 0:
                accuracy = region_correct[in_bin].mean() * 100
            else:
                accuracy = 0.0
            
            percentage = (count / total_count * 100) if total_count > 0 else 0.0
            
            results[region].append({
                'Margin_Range': f'[{low:.1f}, {high:.1f}' + (']' if high == 1.0 else ')'),
                'Count': int(count),
                'Accuracy': accuracy,
                'Percentage': percentage
            })
    
    return results


def print_analysis_table(results: dict, city: str) -> None:
    """Print formatted analysis table to console."""
    print(f"\n{'='*60}")
    print(f"MARGIN ANALYSIS: {city}")
    print(f"{'='*60}")
    
    for region in ['Island', 'Bridge']:
        print(f"\n{region} Nodes:")
        print(f"{'Margin Range':<15} {'Count':>10} {'Accuracy':>12} {'% of Total':>12}")
        print("-" * 51)
        
        for row in results[region]:
            print(f"{row['Margin_Range']:<15} {row['Count']:>10} {row['Accuracy']:>11.2f}% {row['Percentage']:>11.2f}%")
        
        # Print totals
        total_count = sum(r['Count'] for r in results[region])
        total_correct = sum(r['Count'] * r['Accuracy'] / 100 for r in results[region])
        overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0
        print("-" * 51)
        print(f"{'Total':<15} {total_count:>10} {overall_acc:>11.2f}% {100.0:>11.2f}%")


def results_to_dataframe(all_results: dict) -> pd.DataFrame:
    """Convert all results to a DataFrame."""
    rows = []
    for city, results in all_results.items():
        for region in ['Island', 'Bridge']:
            for row in results[region]:
                rows.append({
                    'City': city,
                    'Region': region,
                    'Margin_Range': row['Margin_Range'],
                    'Count': row['Count'],
                    'Accuracy': row['Accuracy'],
                    'Percentage': row['Percentage']
                })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Margin Analysis for Pseudo-label Quality')
    parser.add_argument('--ratio', type=str, default='05pct',
                        choices=['05pct', '10pct', '15pct', '20pct', '25pct'],
                        help='Training ratio (default: 05pct)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='Margin_Analysis_1.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MARGIN ANALYSIS: Testing Margin as Quality Estimator")
    print("="*60)
    print(f"Base Model: MLP")
    print(f"Cities: {CITIES}")
    print(f"Ratio: {args.ratio}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("="*60)
    
    # Store all results
    all_results = {}
    
    for city in CITIES:
        print(f"\nProcessing {city}...")
        
        # Load data
        try:
            data_dict = load_city_data(city, args.ratio)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue
        
        # Get base model predictions (MLP)
        base_probs = get_base_predictions(
            model_name='MLP',
            in_dim=data_dict['in_dim'],
            x=data_dict['x'],
            y=data_dict['y'],
            train_mask=data_dict['train_mask'],
            edge_index=data_dict['edge_index'],
            seed=args.seed
        )
        
        # Analyze margin vs accuracy
        results = compute_margin_analysis(data_dict, base_probs)
        all_results[city] = results
        
        # Print table
        print_analysis_table(results, city)
    
    # Convert to DataFrame and save
    df = results_to_dataframe(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print summary across all cities
    print("\n" + "="*60)
    print("SUMMARY: Average Accuracy by Margin Range")
    print("="*60)
    
    for region in ['Island', 'Bridge']:
        print(f"\n{region}:")
        region_df = df[df['Region'] == region]
        summary = region_df.groupby('Margin_Range').agg({
            'Count': 'sum',
            'Accuracy': 'mean'
        }).reset_index()
        
        # Sort by margin range
        margin_order = ['[0.0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']
        summary['sort_key'] = summary['Margin_Range'].apply(lambda x: margin_order.index(x) if x in margin_order else 99)
        summary = summary.sort_values('sort_key').drop('sort_key', axis=1)
        
        print(f"{'Margin Range':<15} {'Total Count':>12} {'Avg Accuracy':>12}")
        print("-" * 41)
        for _, row in summary.iterrows():
            print(f"{row['Margin_Range']:<15} {row['Count']:>12} {row['Accuracy']:>11.2f}%")


if __name__ == "__main__":
    main()
