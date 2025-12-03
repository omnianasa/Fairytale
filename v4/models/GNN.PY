import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNN(nn.Module):
    """
    Graph Neural Network for Islamic Art feature extraction.
    Features:
    - Multi-scale node embeddings
    - Graph Attention for edge weighting
    - Positional encoding for nodes
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, heads: int = 4):
        super(GNN, self).__init__()
        # Stage 1: Graph Attention Conv
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        # Stage 2: Graph Attention Conv
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        # Final linear to output_dim
        self.fc = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index, pos_encoding=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge list [2, num_edges]
            pos_encoding: Optional positional encoding [num_nodes, pos_dim]
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        if pos_encoding is not None:
            # Concatenate positional encoding to node features
            x = torch.cat([x, pos_encoding], dim=-1)

        # Layer 1
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.gat2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # Final linear layer for embeddings
        x = self.fc(x)
        return x
