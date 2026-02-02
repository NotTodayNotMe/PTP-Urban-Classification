"""
Configuration
=============
Global constants and settings.
"""

import torch

# =============================================================================
# Dataset Configuration
# =============================================================================
CITIES = ["Harbin", "London", "Moscow", "Paris"]
ALL_RATIOS = ["05pct", "10pct", "15pct", "20pct", "25pct"]
DATA_DIR = "Graph_Dataset_Block_Balanced"
NUM_CLASSES = 3

# =============================================================================
# Training Configuration
# =============================================================================
EPOCHS = 1000
PATIENCE = 50
HIDDEN_DIM = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# =============================================================================
# Device Configuration
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
