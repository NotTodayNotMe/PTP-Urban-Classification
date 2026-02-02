"""
Base Models
===========
Contains all base models: MLP, GNNs (GCN, GAT, GraphSAGE, SGC, APPNP, GPRGNN, H2GCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv, APPNP
from torch_geometric.utils import degree


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        return x


class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class SGC(nn.Module):
    """Simple Graph Convolution"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.conv = SGConv(in_dim, out_dim, K=K)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class APPNP_Net(nn.Module):
    """APPNP"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 K: int = 10, alpha: float = 0.1, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x


class GPRGNN(nn.Module):
    """Generalized PageRank GNN"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 K: int = 10, alpha: float = 0.1, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.K = K
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(
            torch.tensor([alpha * (1 - alpha) ** k for k in range(K + 1)])
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        h = self.lin2(x)
        
        row, col = edge_index
        num_nodes = h.size(0)
        
        deg = degree(row, num_nodes, dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        hidden_list = [h]
        for _ in range(self.K):
            h_new = torch.zeros_like(h)
            h_new.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(1)),
                               h[col] * norm.unsqueeze(-1))
            h = h_new
            hidden_list.append(h)
        
        gamma = F.softmax(self.alpha, dim=0)
        out = sum(gamma[k] * hidden_list[k] for k in range(self.K + 1))
        return out


class H2GCN(nn.Module):
    """H2GCN - Heterophily-aware GCN"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 K: int = 2, dropout: float = 0.5):
        super().__init__()
        self.K = K
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim * (1 + 2 * K), out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        
        deg = degree(row, num_nodes, dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        
        h = F.relu(self.lin_in(x))
        h = self.dropout(h)
        
        features = [h]
        h_self = h
        h_neighbor = h
        
        for k in range(self.K):
            h_new = torch.zeros_like(h_neighbor)
            h_new.scatter_add_(0, row.unsqueeze(-1).expand(-1, h_neighbor.size(1)), 
                              h_neighbor[col])
            h_new = h_new * deg_inv.unsqueeze(-1)
            features.append(h_new)
            
            h_higher = torch.zeros_like(h_self)
            h_higher.scatter_add_(0, row.unsqueeze(-1).expand(-1, h_self.size(1)), 
                                 h_self[col])
            h_higher = h_higher * deg_inv.unsqueeze(-1)
            features.append(h_higher)
            
            h_neighbor = h_new
            h_self = h_higher
        
        combined = torch.cat(features, dim=1)
        out = self.lin_out(combined)
        return out


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    'MLP': (MLP, False),           # (class, use_edge_index)
    'GCN': (GCN, True),
    'GAT': (GAT, True),
    'GraphSAGE': (GraphSAGE, True),
    'SGC': (SGC, True),
    'APPNP': (APPNP_Net, True),
    'GPRGNN': (GPRGNN, True),
    'H2GCN': (H2GCN, True),
}
