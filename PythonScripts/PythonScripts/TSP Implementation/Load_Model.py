import torch
from torch.autograd import Variable
import torch.nn as nn
import GNN_Implementation

num_nodes = 10
num_neighbors = -1  # Could increase it!
train_filepath = f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_train_concorde.txt"
hidden_dim = 300
num_layers = 5
mlp_layers = 2
learning_rate = 0.001
max_epochs = 60
batches_per_epoch = 10000

variables = {
    'train_filepath': f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_train_concorde.txt",
    'val_filepath': f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_val_concorde.txt",
    'test_filepath': f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_test_concorde.txt",
    'num_nodes': num_nodes,
    'num_neighbors': num_neighbors,
    'node_dim': 2,
    'voc_nodes_in': 2,
    'voc_nodes_out': 2,
    'voc_edges_in': 3,
    'voc_edges_out': 2,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers,
    'mlp_layers': mlp_layers,
    'aggregation': 'mean',
    'max_epochs': max_epochs,
    'val_every': 5,
    'test_every': 5,
    'batches_per_epoch': batches_per_epoch,
    'accumulation_steps': 1,
    'learning_rate': learning_rate,
    'decay_rate': 1.01
}

model = nn.DataParallel(
    GNN_Implementation.ResidualGatedGCNModel(variables, torch.cuda.FloatTensor, torch.cuda.LongTensor))
model.load_state_dict(torch.load("Models/GraphML_GNN.pth"))
