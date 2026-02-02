"""
Experiment Runner
=================

Main script for running urban region classification experiments.

Usage:
    python run_experiment.py --city London --ratio 05pct --n_runs 1 --imlp_only
    python run_experiment.py --city all --ratio all --n_runs 10

Files in this folder:
    - config.py: Constants and settings
    - data_utils.py: Data loading
    - evaluation.py: Metrics computation
    - base_models.py: GNN and ML models
    - trainer.py: Model training
    - post_processing.py: LP, C&S, IM-LP (modify this for new methods)
    - run_experiment.py: This file (experiment runner)
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# Import from local files
from config import CITIES, ALL_RATIOS, NUM_CLASSES, device
from data_utils import load_city_data, get_data_info, get_train_test_masks
from evaluation import compute_metrics
from trainer import get_base_predictions
from post_processing import (
    IMLP, IMLPConfig,
    LabelPropagation, LPConfig,
    CorrectAndSmooth, CSConfig
)


# =============================================================================
# Experiment Configuration
# =============================================================================

BASE_MODELS = ['RF', 'MLP', 'GCN', 'GAT', 'GraphSAGE', 'SGC', 'APPNP', 'GPRGNN', 'H2GCN']

POST_METHODS = {
    'None': None,
    'LP': LabelPropagation(LPConfig(num_iter=50, alpha=0.9)),
    'CS': CorrectAndSmooth(CSConfig()),
    'IMLP': IMLP(IMLPConfig()),
}


# =============================================================================
# Experiment Functions
# =============================================================================

def run_single_experiment(data_dict: dict, 
                          base_model: str,
                          post_method: str,
                          seed: int) -> dict:
    """Run a single experiment with one base model and post-processing method."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    x = data_dict['x']
    y = data_dict['y']
    edge_index = data_dict['edge_index']
    train_mask = data_dict['train_mask']
    num_nodes = data_dict['num_nodes']
    in_dim = data_dict['in_dim']
    
    island_test_mask, bridge_test_mask = get_train_test_masks(data_dict)
    
    # Get base predictions
    base_probs = get_base_predictions(
        base_model, in_dim, x, y, train_mask, edge_index, seed
    )
    
    # Apply post-processing
    if post_method == 'None' or POST_METHODS[post_method] is None:
        final_probs = base_probs
    else:
        method = POST_METHODS[post_method]
        final_probs = method(base_probs, edge_index, y, train_mask, num_nodes, NUM_CLASSES)
    
    # Compute metrics
    results = {
        'Island': compute_metrics(final_probs, y, island_test_mask),
        'Bridge': compute_metrics(final_probs, y, bridge_test_mask)
    }
    
    return results


def run_full_experiment(city: str,
                        ratio: str,
                        n_runs: int = 10,
                        base_models: list = None,
                        post_methods: list = None,
                        output_dir: str = 'results') -> pd.DataFrame:
    """Run full experiment for a city and ratio."""
    
    if base_models is None:
        base_models = BASE_MODELS
    if post_methods is None:
        post_methods = list(POST_METHODS.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"City: {city}, Ratio: {ratio}")
    print(f"{'='*60}")
    
    try:
        data_dict = load_city_data(city, ratio)
        print(get_data_info(data_dict))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    all_rows = []
    seeds = []
    
    for run in range(n_runs):
        seed = 42 + run
        seeds.append(seed)
        print(f"\n  Run {run+1}/{n_runs} (seed={seed})...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for base_model in base_models:
            for post_method in post_methods:
                method_name = f"{base_model}" if post_method == 'None' else f"{base_model}->{post_method}"
                
                results = run_single_experiment(data_dict, base_model, post_method, seed)
                
                for region in ['Island', 'Bridge']:
                    row = {
                        'City': city,
                        'Ratio': ratio,
                        'Base': base_model,
                        'Post': post_method,
                        'Method': method_name,
                        'Run': run + 1,
                        'Region': region,
                        **results[region]
                    }
                    all_rows.append(row)
                
                if post_method == 'IMLP':
                    print(f"    {method_name}: Bridge F1={results['Bridge']['Macro_F1']:.4f}")
    
    df = pd.DataFrame(all_rows)
    
    filename = os.path.join(output_dir, f"results_{city}_{ratio}.csv")
    df.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")
    
    seeds_file = os.path.join(output_dir, f"seeds_{city}_{ratio}.json")
    with open(seeds_file, 'w') as f:
        json.dump({'seeds': seeds}, f)
    
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary of results."""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print("SUMMARY (Bridge Macro-F1)")
    print(f"{'='*80}")
    
    summary = df[df['Region'] == 'Bridge'].groupby(['Method'])['Macro_F1'].agg(['mean', 'std'])
    summary = summary.sort_values('mean', ascending=False)
    
    print(f"\n{'Method':<25} {'Macro-F1':>15}")
    print("-" * 42)
    for method, row in summary.iterrows():
        print(f"  {method:<23} {row['mean']:.4f}±{row['std']:.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Urban Region Classification Experiment')
    parser.add_argument('--city', type=str, default='all',
                        choices=['Harbin', 'London', 'Moscow', 'Paris', 'all'])
    parser.add_argument('--ratio', type=str, default='05pct',
                        choices=['05pct', '10pct', '15pct', '20pct', '25pct', 'all'])
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--base_only', action='store_true',
                        help='Run only base models (no post-processing)')
    parser.add_argument('--imlp_only', action='store_true',
                        help='Run only IMLP post-processing')
    
    args = parser.parse_args()
    
    cities = CITIES if args.city == 'all' else [args.city]
    ratios = ALL_RATIOS if args.ratio == 'all' else [args.ratio]
    
    if args.base_only:
        post_methods = ['None']
    elif args.imlp_only:
        post_methods = ['None', 'IMLP']
    else:
        post_methods = list(POST_METHODS.keys())
    
    print("="*80)
    print("URBAN REGION CLASSIFICATION EXPERIMENT")
    print("="*80)
    print(f"Cities: {cities}")
    print(f"Ratios: {ratios}")
    print(f"Runs: {args.n_runs}")
    print(f"Post-processing: {post_methods}")
    print(f"Device: {device}")
    print("="*80)
    
    all_dfs = []
    for city in cities:
        for ratio in ratios:
            df = run_full_experiment(
                city, ratio,
                n_runs=args.n_runs,
                post_methods=post_methods,
                output_dir=args.output_dir
            )
            if df is not None:
                all_dfs.append(df)
                print_summary(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_file = os.path.join(args.output_dir, "all_results.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"\nSaved combined results: {combined_file}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
