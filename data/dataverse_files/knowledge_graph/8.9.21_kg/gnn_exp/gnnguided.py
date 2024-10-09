import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from torch.nn import Linear
import os
import json
from sklearn.model_selection import train_test_split
from typing import Dict, Any

class GuidedGNN(torch.nn.Module):
    def __init__(self, features_size, embedding_size=2048, heads=3, dropout=0.2):
        super(GuidedGNN, self).__init__()
        # GATConv layers
        self.conv1 = GATConv(features_size, embedding_size, heads=heads, dropout=dropout)
        self.conv2 = GATConv(embedding_size * heads, embedding_size, heads=heads, dropout=dropout)
        self.conv3 = GATConv(embedding_size * heads, embedding_size, heads=heads, dropout=dropout)
        
        # Linear transformation layers to project embeddings
        self.head_transform1 = Linear(embedding_size * heads, embedding_size)
        self.head_transform2 = Linear(embedding_size * heads, embedding_size)
        self.head_transform3 = Linear(embedding_size * heads, embedding_size)
        
        # Final layers for classification
        self.linear1 = Linear(embedding_size, 512)
        self.linear2 = Linear(512, 1)

    def forward(self, x, edge_index):
        # GATConv layers with ELU activation
        x = F.elu(self.conv1(x, edge_index))
        x = self.head_transform1(x)
        
        x = F.elu(self.conv2(x, edge_index))
        x = self.head_transform2(x)
        
        x = F.elu(self.conv3(x, edge_index))
        x = self.head_transform3(x)
        
        # Linear layers for classification
        x = F.elu(self.linear1(x))
        scores = self.linear2(x).squeeze(-1)
        return torch.sigmoid(scores)


def load_patient_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)


patient_files = [
    f"./patient_subgraph_data/labeled_patient_data/patient_{i}_result.json" 
    for i in range(4000) 
    if os.path.exists(f"./patient_subgraph_data/labeled_patient_data/patient_{i}_result.json")
]

print(len(patient_files))
train_files, test_files = train_test_split(patient_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")


def train_guided_gnn(data, patient_files, model, optimizer, criterion, top_k=2500, threshold=0.5):
    model.train()
    total_loss = 0

    for patient_file in patient_files:
        # Load patient-specific data
        patient = load_patient_data(patient_file)
        phenotype_indices = patient['positive_phenotypes']['indices']
        true_gene_index = patient['true_gene']['index']

        # Activate phenotypes by adding 1 to their node features
        x = torch.cat([data[node_type].x for node_type in data.node_types], dim=0).clone()
        edge_index = torch.cat([data[edge_type].edge_index for edge_type in data.edge_types], dim=1)

        x[phenotype_indices] += 1

        # Forward pass through GuidedGNN
        scores = model(x, edge_index)

        # Retrieve top-k nodes based on scores
        _, top_k_indices = torch.topk(scores, k=top_k)
        relevant_nodes = top_k_indices[scores[top_k_indices] > threshold]

        # Check if true gene is in the retrieved subgraph
        if true_gene_index in relevant_nodes:
            loss = 0  # Perfect prediction
        else:
            loss = criterion(scores, torch.zeros_like(scores))  # Penalize for missing true gene

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(patient_files)


# Example Usage
features_size = 2048
embedding_size = 2048
heads = 3
dropout = 0.2

# Define the model
model = GuidedGNN(features_size, embedding_size, heads, dropout)

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# Load the pretrained graph data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = torch.load('./KGs/Shepherd_KG_with_pretrained_embeddings3.pt')
print("Graph loaded from 'Shepherd_KG_with_pretrained_embeddings3.pt'.")

# Move data to device
data = data.to(device)

# Train the model
loss = train_guided_gnn(data, train_files, model.to(device), optimizer, criterion)
print(f"Training loss: {loss}")