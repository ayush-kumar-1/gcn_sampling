import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.loader import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler

import tqdm

def main(): 
    dataset = Planetoid(root='../data/', name='PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    #for reproducibility 
    torch.manual_seed(12345)
    np.random.seed(12345)

    train_loader = GraphSAINTRandomWalkSampler(data, batch_size=128, num_steps=100, walk_length=20)  # 2. Stochastic partioning scheme.
    device = "cpu"

    model = GAT(dataset).to(device)
    data = dataset[0].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train(): 
        model.train()
        
        for sub_data in train_loader: 
            out = model(sub_data)
            loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
    def test():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
            accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
        return accs
            
    t = tqdm.trange(1, 100, desc='Epoch 1')
    for epoch in t:
        loss = train()
        train_acc, val_acc, test_acc = test()
        t.set_description(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')


class GAT(torch.nn.Module):
    def __init__(self, dataset):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                            heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    main()