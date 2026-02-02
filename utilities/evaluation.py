"""
Evaluation Utilities
====================
Functions for computing evaluation metrics.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from config import NUM_CLASSES


def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute prediction margin (top1 - top2 probability).
    
    Args:
        probs: Probability tensor of shape (N, C)
    
    Returns:
        Margin tensor of shape (N,)
    """
    sorted_p = torch.sort(probs, dim=1, descending=True)[0]
    return sorted_p[:, 0] - sorted_p[:, 1]


def compute_metrics(probs: torch.Tensor, 
                    y: torch.Tensor, 
                    mask: torch.Tensor, 
                    num_classes: int = NUM_CLASSES) -> dict:
    """
    Compute evaluation metrics: AUC, Recall, Macro-F1, Micro-F1.
    
    Args:
        probs: Probability predictions (N, C)
        y: Ground truth labels (N,)
        mask: Boolean mask for nodes to evaluate (N,)
        num_classes: Number of classes
    
    Returns:
        Dictionary with AUC, Recall, Macro_F1, Micro_F1
    """
    pred = probs.argmax(dim=1)
    probs_np = probs.cpu().numpy()
    y_np = y.cpu().numpy()
    pred_np = pred.cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    valid = mask_np & (y_np >= 0)
    
    if valid.sum() == 0:
        return {'AUC': 0.0, 'Recall': 0.0, 'Macro_F1': 0.0, 'Micro_F1': 0.0}
    
    y_valid = y_np[valid]
    pred_valid = pred_np[valid]
    probs_valid = probs_np[valid]
    
    # F1 scores
    macro_f1 = f1_score(y_valid, pred_valid, average='macro', zero_division=0)
    micro_f1 = f1_score(y_valid, pred_valid, average='micro', zero_division=0)
    
    # Recall
    recall = recall_score(y_valid, pred_valid, average='macro', zero_division=0)
    
    # AUC
    try:
        y_bin = label_binarize(y_valid, classes=list(range(num_classes)))
        if y_bin.shape[1] == num_classes:
            auc = roc_auc_score(y_bin, probs_valid, average='macro', multi_class='ovr')
        else:
            auc = 0.5
    except:
        auc = 0.5
    
    return {
        'AUC': float(auc), 
        'Recall': float(recall), 
        'Macro_F1': float(macro_f1), 
        'Micro_F1': float(micro_f1)
    }
