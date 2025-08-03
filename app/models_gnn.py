import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, GINConv,GATConv, GraphConv, global_max_pool, global_mean_pool
from torch.nn import Linear, Dropout, GELU, BatchNorm1d

class HybridGNN(nn.Module):
    def __init__(self, hidden_channels, descriptor_size, dropout_rate):
        super(HybridGNN, self).__init__()

        self.conv1 = GINConv(Linear(9, hidden_channels))
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

        self.gelu = nn.GELU()
        self.dropout = Dropout(p=dropout_rate)

        combined_size = hidden_channels * 2 + descriptor_size

        self.shared_fc = nn.Sequential(
            Linear(combined_size, 128),
            nn.GELU(),
            Dropout(dropout_rate)
        )

        self.tg_head = nn.Sequential(
            Linear(128, 64),
            nn.GELU(),
            Dropout(dropout_rate),
            Linear(64, 1)
        )

        self.tm_head = nn.Sequential(
            Linear(128, 64),
            nn.GELU(),
            Dropout(dropout_rate),
            Linear(64, 1)
        )

    def forward(self, x, edge_index, batch_index, descriptors):
        h = self.gelu(self.conv1(x, edge_index))
        h = self.gelu(self.conv2(h, edge_index))
        h = self.gelu(self.conv3(h, edge_index))

        h = torch.cat([
            global_max_pool(h, batch_index),
            global_mean_pool(h, batch_index)
        ], dim=1)

        combined = torch.cat([h, descriptors], dim=1)
        shared_features = self.shared_fc(combined)

        tg_out = self.tg_head(shared_features)
        tm_out = self.tm_head(shared_features)
        return torch.cat([tg_out, tm_out], dim=1), shared_features

class SolubilityGNN(nn.Module):
    def __init__(self, hidden_channels, descriptor_size, dropout_rate):
        super(SolubilityGNN, self).__init__()

        # TransformerConv layers
        self.trans1 = TransformerConv(in_channels=9, out_channels=hidden_channels, heads=4, concat=True, edge_dim=6)
        self.bn1 = BatchNorm1d(hidden_channels * 4)

        self.trans2 = TransformerConv(in_channels=hidden_channels * 4, out_channels=hidden_channels, heads=4, concat=True, edge_dim=6)
        self.bn2 = BatchNorm1d(hidden_channels * 4)

        # GIN layer (does not use edge_attr)
        self.gin3 = GINConv(Linear(hidden_channels * 4, hidden_channels))
        self.bn3 = BatchNorm1d(hidden_channels)

        # Fully connected layers
        self.fc1 = Linear(hidden_channels * 4 + descriptor_size * 2, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 32)
        self.out = Linear(32, 1)

        self.gelu = GELU()
        self.dropout = Dropout(p=dropout_rate)

    def gnn_block(self, x, edge_index, edge_attr):
        # Transformer layers with edge features
        x = self.gelu(self.bn1(self.trans1(x, edge_index, edge_attr)))
        x = self.dropout(x)

        x = self.gelu(self.bn2(self.trans2(x, edge_index, edge_attr)))
        x = self.dropout(x)

        # GINConv doesn't use edge_attr
        x = self.gelu(self.bn3(self.gin3(x, edge_index)))
        x = self.dropout(x)

        return x

    def forward(self, x1, edge_index1, batch_index1, descriptors1, edge_attr1,
                      x2, edge_index2, batch_index2, descriptors2, edge_attr2):

        # GNN block for molecule 1
        h1 = self.gnn_block(x1, edge_index1, edge_attr1)
        h1 = torch.cat([
            global_max_pool(h1, batch_index1),
            global_mean_pool(h1, batch_index1)
        ], dim=1)

        # GNN block for molecule 2
        h2 = self.gnn_block(x2, edge_index2, edge_attr2)
        h2 = torch.cat([
            global_max_pool(h2, batch_index2),
            global_mean_pool(h2, batch_index2)
        ], dim=1)

        # Combine pooled features with descriptors
        hybrid_features = torch.cat([h1, h2, descriptors1, descriptors2], dim=1)

        # Fully connected head
        x = self.gelu(self.fc1(hybrid_features))
        x = self.dropout(x)

        x = self.gelu(self.fc2(x))
        x = self.dropout(x)

        x = self.gelu(self.fc3(x))
        x = self.dropout(x)

        out = self.out(x)
        return out, hybrid_features
