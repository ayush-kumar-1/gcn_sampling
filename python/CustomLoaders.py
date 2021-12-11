import numpy as np 
import tensorflow as tf
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix

from spektral.data import Loader
from spektral.utils import (batch_generator, 
    batch_generator,
    collate_labels_disjoint,
    get_spec,
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_batch,
    to_disjoint,
    to_mixed,
    to_tf_signature,
)

class GraphSAGELoader(Loader): 
    """
    The GraphSAGE mini-batch algorithm uses k-hop neighbors specified at a depth k, 
    to create a mini-batch of nodes. Only works for a single graph.
    
    "For each node in the training graph the sampling method samples k-hop neighbors by search depth. 
    Then the sampled neighbors are added to the minibtach node set B^k for storage." (Liu Et. Al)

    This is a implementation of the GraphSAGE minibatch sampling method for Spektral. 
    It is based on the paper: Inductive Representation Learning on Large Graphs (Hamilton, Ying & Leskovec 2017)
    """

    def __init__(self, dataset, 
        epochs=None, 
        num_hops=2,
        max_neighbors=4,
        shuffle=True): 
        """
        Args: 
            dataset: A spektral.data.Dataset object. 
            epochs: The number of epochs to run the loader. 
            sample_weights: A list of weights to be applied to each sample in the dataset. 
        """
        batch_size = (max_neighbors + 1) * num_hops
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def load(): 
        """
        Loads the next batch of graphs. 
        """
        return self.collate(self.assemble_dataset_from_sample(graph, sample))

    def collate(self, batch): 
        """
        Takes as input a list of Graph objects and returns a lit of Tensors, np.arrays, 
        or Sparse Tensors.  
        """
        return batch

    def assemble_dataset_from_sample(graph, sample):
        """
        Assembles a dataset from a sample of nodes.
        """
        nGraph = from_numpy_matrix(graph.a)
        subgraph = sGraph(a = to_numpy_matrix(nGraph.subgraph(sample)), 
                    x = graph.x[sample], 
                    y = graph.y[sample])
        return subgraph
        

    def k_hop_sample(max_k = 2, seed_node = 0, N = 4): 
        return k_hop_helper(k = 0, max_k = max_k, seed_nodes=[seed_node], N = N)
            
    def k_hop_helper(k, max_k, seed_nodes, N = 4): 
        if k == max_k: 
            return seed_nodes 
        
        chosen_nodes = []
        chosen_nodes += seed_nodes
        for seed_node in seed_nodes: 
            neighbors =  [node for node in networkx_graph.neighbors(seed_node)]
            if len(neighbors) >= N: 
                chosen_nodes += [*np.random.choice(neighbors, N, replace=False)]
            else: 
                q, r = divmod(N*2, len(neighbors_1))
                neighbors = q * neighbors + neighbors[:r]
                chosen_nodes += [*np.random.choice(neighbors, N, replace=True)] 
    
        return k_hop_helper(k = k+1, max_k=max_k, seed_nodes = chosen_nodes, N=N)
    
    @property
    def steps_per_epoch(self):
        return self.dataset[0].n_nodes
    

    
    
