"""
Data Utilities
==============
Functions for loading and preprocessing city graph datasets.
"""

import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, device


def load_city_data(city: str, ratio: str, data_dir: str = None):
    """
    Load and preprocess city graph data.
    
    Args:
        city: City name (Harbin, London, Moscow, Paris)
        ratio: Training ratio (05pct, 10pct, 15pct, 20pct, 25pct)
        data_dir: Directory containing data files (optional)
    
    Returns:
        dict containing:
            - x: Node features (normalized, on device)
            - y: Node labels (on device)
            - edge_index: Graph edges (on device)
            - train_mask: Training mask (on device)
            - test_mask: Test mask (on device)
            - bridge_mask: Bridge node mask (on device)
            - num_nodes: Number of nodes
            - in_dim: Input feature dimension
            - data: Original PyG data object
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    data_path = os.path.join(data_dir, f"{city}_{ratio}.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # Normalize features using training data statistics
    X = data.x.numpy()
    scaler = StandardScaler()
    scaler.fit(X[data.train_mask.numpy()])
    X = scaler.transform(X)
    
    # Move to device
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)
    bridge_mask = data.global_bridge_mask.to(device)
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'bridge_mask': bridge_mask,
        'num_nodes': x.size(0),
        'in_dim': x.size(1),
        'data': data
    }


def get_data_info(data_dict: dict) -> str:
    """Get a formatted string with dataset information."""
    num_nodes = data_dict['num_nodes']
    num_train = data_dict['train_mask'].sum().item()
    num_test = data_dict['test_mask'].sum().item()
    num_bridge = data_dict['bridge_mask'].sum().item()
    in_dim = data_dict['in_dim']
    num_edges = data_dict['edge_index'].size(1)
    
    info = (f"Nodes: {num_nodes:,}, Edges: {num_edges:,}, "
            f"Train: {num_train:,}, Bridge: {num_bridge:,}")
    return info


def get_train_test_masks(data_dict: dict):
    """
    Get separate masks for Island and Bridge test nodes.
    
    Returns:
        tuple: (island_test_mask, bridge_test_mask)
    """
    test_mask = data_dict['test_mask']
    bridge_mask = data_dict['bridge_mask']
    
    island_test_mask = test_mask & ~bridge_mask
    bridge_test_mask = test_mask & bridge_mask
    
    return island_test_mask, bridge_test_mask
