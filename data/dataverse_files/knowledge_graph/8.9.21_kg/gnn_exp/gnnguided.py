import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch.nn import Linear, ModuleDict
import os
import json
from sklearn.model_selection import train_test_split
from typing import Dict, Any
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GuidedGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATConv((-1, -1), hidden_channels, add_self_loops=False)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = ModuleDict({
            node_type: Linear(hidden_channels, out_channels) 
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        return {key: self.lin[key](x).squeeze(-1) for key, x in x_dict.items()}

def load_patient_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def train_guided_gnn(data, patient_files, model, optimizer, criterion, top_k=2500, threshold=0.5, batch_size=32):
    model.train()
    total_loss = 0
    total_processed = 0

    loader = NeighborLoader(
        data,
        num_neighbors={key: [10] * 2 for key in data.edge_types},
        batch_size=batch_size,
        input_nodes=('gene/protein', None),
    )

    pbar = tqdm(total=len(patient_files), desc="Training Progress")

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        batch_loss = 0

        for patient_file in patient_files:
            patient = load_patient_data(patient_file)
            phenotype_indices = patient['positive_phenotypes']['indices']
            true_gene_index = patient['true_gene']['index']

            x_dict = {node_type: batch[node_type].x.clone() for node_type in batch.node_types}
            edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}

            if 'effect/phenotype' in x_dict:
                global_to_batch = {idx.item(): i for i, idx in enumerate(batch['effect/phenotype'].original_node_ids)}
                batch_phenotype_indices = torch.tensor([global_to_batch[idx] for idx in phenotype_indices if idx in global_to_batch])
                if len(batch_phenotype_indices) > 0:
                    x_dict['effect/phenotype'][batch_phenotype_indices] += 1

            out_dict = model(x_dict, edge_index_dict)
            scores = out_dict['gene/protein']

            gene_global_to_batch = {idx.item(): i for i, idx in enumerate(batch['gene/protein'].original_node_ids)}
            if true_gene_index in gene_global_to_batch:
                batch_true_gene_index = gene_global_to_batch[true_gene_index]

                actual_k = min(top_k, scores.numel())

                if actual_k > 0:
                    _, top_k_indices = torch.topk(scores, k=actual_k)
                    relevant_nodes = top_k_indices[scores[top_k_indices] > threshold]

                    if batch_true_gene_index in relevant_nodes:
                        loss = torch.tensor(0.0, device=device)
                    else:
                        loss = criterion(scores, torch.zeros_like(scores))

                    batch_loss += loss
                    total_processed += 1
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)

        if batch_loss > 0:
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

    pbar.close()
    
    if total_processed > 0:
        avg_loss = total_loss / total_processed
        logger.info(f"Average training loss: {avg_loss:.4f}")
        return avg_loss
    else:
        logger.warning("No patient files were processed successfully.")
        return None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    data = torch.load('./TEST1.pt')
    logger.info("Graph loaded from 'TEST1.pt'.")

    data = data.to(device)

    patient_files = [
        f"./patient_subgraph_data/labeled_patient_data/patient_{i}_result.json" 
        for i in range(200) 
        if os.path.exists(f"./patient_subgraph_data/labeled_patient_data/patient_{i}_result.json")
    ]

    logger.info(f"Total patient files found: {len(patient_files)}")
    train_files, test_files = train_test_split(patient_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    logger.info(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

    hidden_channels = 64
    out_channels = 1
    num_layers = 2

    model = GuidedGNN(data.metadata(), hidden_channels, out_channels, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss = train_guided_gnn(data, train_files, model, optimizer, criterion)
    logger.info(f"Final training loss: {loss:.4f}")