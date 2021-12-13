import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.loader import RandomNodeSampler

import tqdm

def main(): 
    baseline()
    
    # graphsaint_RandomNodeSampler()
    dataset = Planetoid(root='../data/', name='PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]

    random_loader = RandomNodeSampler(data, num_parts=10)
    sampling(random_loader, "RandomNodeSampler")

    cluster_data = ClusterData(data, num_parts=32)
    cluster_loader = ClusterLoader(cluster_data, batch_size=128, shuffle=True, num_workers=4)
    sampling(cluster_loader, "ClusterGCN")

    neighbor_loader = NeighborLoader(data, num_neighbors=[20]*2, batch_size=128, input_nodes = data.train_mask)
    sampling(neighbor_loader, "GraphSAGE")
    
    gsaint_node_sampler = GraphSAINTNodeSampler(data, batch_size=32, num_steps=100)
    sampling(gsaint_node_sampler, "GraphSAINTNodeSampler")

    gsaint_edge_sampler = GraphSAINTEdgeSampler(data, batch_size=32, num_steps=100)
    sampling(gsaint_edge_sampler, "GraphSAINTEdgeSampler")

    gsaint_random_walk_sampler = GraphSAINTRandomWalkSampler(data, batch_size=32,  num_steps=100, walk_length=32)
    sampling(gsaint_random_walk_sampler, "GraphSAINTRandomWalkSampler")

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

def baseline(): 
    dataset = Planetoid(root='../data/', name='PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    #for reproducibility 
    #torch.manual_seed(12345)
    #np.random.seed(12345)

    device = "cpu"
    model = GAT(dataset).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def test():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
            accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
        return accs

    print("Baseline (No Sampling):")
    model.train()
    t = tqdm.trange(1, 1000, desc='Epoch 1')
    for epoch in t:
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_acc, val_acc, test_acc = test()
        t.set_description(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        loss.backward()
        optimizer.step()

def sampling(sampling_method, method_name): 
    dataset = Planetoid(root='../data/', name='PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    #for reproducibility 
    #torch.manual_seed(12345)
    #np.random.seed(12345)

    device = "cpu"
    model = GAT(dataset).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train(): 
        model.train()
        
        for sub_data in sampling_method: 
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

    print(f"{method_name} (Sampling):")
    t = tqdm.trange(1, 100, desc='Epoch 1')
    for epoch in t:
        loss = train()
        train_acc, val_acc, test_acc = test()
        t.set_description(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()