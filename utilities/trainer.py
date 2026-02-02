"""
Model Training
==============
Functions for training base models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier

from config import device, EPOCHS, PATIENCE, HIDDEN_DIM, NUM_CLASSES
from base_models import MODEL_REGISTRY, MLP, GCN, GAT, GraphSAGE, SGC, APPNP_Net, GPRGNN, H2GCN


def train_nn_model(model: torch.nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   train_mask: torch.Tensor,
                   edge_index: torch.Tensor,
                   epochs: int = EPOCHS,
                   patience: int = PATIENCE,
                   use_edge: bool = True,
                   lr: float = 0.01,
                   weight_decay: float = 5e-4) -> Tuple[torch.nn.Module, torch.Tensor]:
    """
    Train a neural network model with early stopping.
    
    Returns:
        tuple: (trained_model, probability_predictions)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    valid_train = train_mask & (y >= 0)
    
    best_model_state = None
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if use_edge:
            logits = model(x, edge_index)
        else:
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
    
    model.eval()
    with torch.no_grad():
        if use_edge:
            logits = model(x, edge_index)
        else:
            logits = model(x)
        probs = F.softmax(logits, dim=1)
    
    return model, probs


def train_rf_model(x: torch.Tensor,
                   y: torch.Tensor,
                   train_mask: torch.Tensor,
                   seed: int) -> Tuple[object, torch.Tensor]:
    """
    Train a Random Forest model.
    
    Returns:
        tuple: (trained_model, probability_predictions)
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()
    
    valid_train = train_mask_np & (y_np >= 0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(x_np[valid_train], y_np[valid_train])
    
    probs = model.predict_proba(x_np)
    probs = torch.tensor(probs, dtype=torch.float32, device=device)
    
    return model, probs


def get_base_predictions(model_name: str,
                         in_dim: int,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train_mask: torch.Tensor,
                         edge_index: torch.Tensor,
                         seed: int,
                         hidden_dim: int = HIDDEN_DIM,
                         num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Get predictions from a base model.
    
    Args:
        model_name: Name of the model ('RF', 'MLP', 'GCN', etc.)
        in_dim: Input feature dimension
        x: Node features
        y: Node labels
        train_mask: Training mask
        edge_index: Graph edges
        seed: Random seed
    
    Returns:
        Probability predictions (N, C)
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    if model_name == 'RF':
        _, probs = train_rf_model(x, y, train_mask, seed)
        return probs
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class, use_edge = MODEL_REGISTRY[model_name]
    model = model_class(in_dim, hidden_dim, num_classes)
    
    _, probs = train_nn_model(model, x, y, train_mask, edge_index, use_edge=use_edge)
    
    return probs
