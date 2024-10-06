import torch
import torch_geometric
import torch_geometric.utils as utils
import pandas as pd
import json
import pickle
import os
import argparse
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected

import random

# Load data or graph
try:
    print("Trying to load the knowledge graph from file...")
    with open('knowledge_graph_to.pt', 'rb') as f:
        data = torch.load(f).to('cuda')
    print("Knowledge graph loaded successfully.")

    # Load data for mapping node names to indices
    print("Loading node and edge data for node mapping...")
    nodes_df = pd.read_csv('../KG_node_map_test.csv')
    edges_df = pd.read_csv('../KG_edgelist_mask_test.csv')

    # Create edge attributes
    relation_types = edges_df['full_relation'].unique()
    relation_dict = {relation: idx for idx, relation in enumerate(relation_types)}
except FileNotFoundError:
    print("Knowledge graph file not found. Creating the knowledge graph...")
    # Load data
    print("Loading node and edge data...")
    nodes_df = pd.read_csv('../KG_node_map_test.csv')
    edges_df = pd.read_csv('../KG_edgelist_mask_test.csv')

    # Extract node and edge data
    node_features = torch.tensor(pd.factorize(nodes_df['node_idx'])[0], dtype=torch.long).to('cuda')
    node_types = torch.tensor(pd.factorize(nodes_df['node_type'])[0], dtype=torch.long).to('cuda')
    edge_index = torch.tensor(edges_df[['x_idx', 'y_idx']].values.T, dtype=torch.long).to('cuda')

    # Create edge attributes
    relation_types = edges_df['full_relation'].unique()
    relation_dict = {relation: idx for idx, relation in enumerate(relation_types)}
    edge_attr = torch.tensor([relation_dict[relation] for relation in edges_df['full_relation']], dtype=torch.long).to('cuda')

    # Create a graph using PyG's Data
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    data = ToUndirected()(data)

    # Save the knowledge graph for future use
    with open('knowledge_graph_to.pt', 'wb') as f:
        torch.save(data, f)
    data = data.to('cuda')
    print("Knowledge graph saved to file.")

# Map node names to indices
print("Mapping node names to indices...")
node_name_to_idx = {row['node_name']: row['node_idx'] for _, row in nodes_df.iterrows()}

# Load patient data from file
def load_patients(file_path, max_samples=None):
    print(f"Loading patients from {file_path}...")
    with open(file_path, 'r') as f:
        patients = json.load(f)
    
    total_patients = len(patients)
    if max_samples is not None and max_samples < total_patients:
        patients = patients[:max_samples]
        print(f"Loaded the first {max_samples} patients out of {total_patients}.")
    else:
        print(f"Loaded all {total_patients} patients.")
    
    return patients

# Function to map phenotypes to node indices
def map_phenotypes_to_indices(phenotype_names, node_name_to_idx):
    indices = []
    for name in phenotype_names:
        idx = node_name_to_idx.get(name)
        if idx is not None:
            indices.append(idx)
        else:
            print(f"Phenotype '{name}' not found in KG.")
    print(f"Mapped phenotypes to indices: {indices}")
    return indices

# Function to map gene to index
def map_gene_to_index(gene_name, node_name_to_idx):
    idx = node_name_to_idx.get(gene_name)
    if idx is None:
        print(f"Gene '{gene_name}' not found in KG.")
    else:
        print(f"Mapped gene '{gene_name}' to index: {idx}")
    return idx

# Function to compute personalized PageRank using power iteration
def compute_ppr_gpu(data, personalization_nodes, alpha=0.15, max_iter=50, tol=1e-6):
    print(f"Computing personalized PageRank for nodes: {personalization_nodes} with alpha={alpha} using GPU...")
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # Initialize teleportation vector and PageRank scores
    personalization = torch.zeros(num_nodes, device='cuda')
    personalization[personalization_nodes] = 1.0 / len(personalization_nodes)
    ppr_scores = torch.ones(num_nodes, device='cuda') / num_nodes

    # Sparse adjacency matrix
    row, col = edge_index
    adj_matrix = torch.sparse_coo_tensor(torch.stack([row, col]), torch.ones(row.size(0), device='cuda'), (num_nodes, num_nodes))
    deg = torch.sparse.sum(adj_matrix, dim=1).to_dense()
    deg_inv = torch.pow(deg, -1)
    deg_inv[deg_inv == float('inf')] = 0

    # Power iteration without explicitly creating large dense matrices
    for i in range(max_iter):
        prev_ppr = ppr_scores.clone()
        adj_normalized = torch.sparse_coo_tensor(adj_matrix._indices(), deg_inv[adj_matrix._indices()[0]] * adj_matrix._values(), adj_matrix.size()).to('cuda')
        ppr_scores = (1 - alpha) * torch.sparse.mm(adj_normalized, ppr_scores.unsqueeze(1)).squeeze() + alpha * personalization
        torch.cuda.empty_cache()  # Free up memory
        if torch.norm(ppr_scores - prev_ppr, p=1) < tol:
            break

    return ppr_scores

# Function to extract subgraph based on PPR scores
def extract_subgraph(ppr_scores, data, top_k):
    print(f"Extracting subgraph with top {top_k} PPR scores...")
    top_nodes = torch.topk(ppr_scores, top_k).indices.flatten()
    subgraph = utils.k_hop_subgraph(top_nodes, num_hops=1, edge_index=data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    return subgraph

# Function to extract triplets from subgraph
def extract_triplets_from_subgraph(subgraph, relation_dict):
    print("Extracting triplets from subgraph...")
    triplets = []
    edge_index, edge_attr = subgraph[1], subgraph[3]
    if edge_attr is not None and edge_index.size(1) == edge_attr.size(0):
        for i in range(edge_index.shape[1]):
            source, target = edge_index[:, i]
            relation_numeric = edge_attr[i].item()
            triplets.append((source.item(), relation_numeric, target.item()))
    else:
        print("Mismatch in edge and attribute sizes. Skipping triplet extraction.")
    print(f"Extracted triplets length: {len(triplets)}")
    return triplets

# Function to process each patient
def process_patient(patient_data, data, node_name_to_idx, relation_dict, top_k_values):
    print(f"Processing patient with ID: {patient_data['id']}...")
    phenotype_indices = map_phenotypes_to_indices(patient_data['positive_phenotypes'], node_name_to_idx)
    true_gene_idx = map_gene_to_index(patient_data['true_genes'][0], node_name_to_idx)

    if true_gene_idx is None:
        print(f"True gene not found in KG for patient {patient_data['id']}")
        return None

    for top_k in top_k_values:
        print(f"Trying top_k={top_k}...")
        # Compute PPR scores using GPU
        ppr_scores = compute_ppr_gpu(data, phenotype_indices)
        
        # Extract subgraph
        subgraph = extract_subgraph(ppr_scores, data, top_k=top_k)
        
        # Check if true gene is in subgraph
        if true_gene_idx in subgraph[0]:
            print(f"True gene {true_gene_idx} found in subgraph with top_k={top_k}.")
            print("\n \n FOUND \n \n")
            # Extract triplets from subgraph
            triplets = extract_triplets_from_subgraph(subgraph, relation_dict)
            patient_result = {
                'patient_id': patient_data['id'],
                'positive_phenotypes': {
                    'names': patient_data['positive_phenotypes'],
                    'indices': phenotype_indices
                },
                'true_gene': {
                    'name': patient_data['true_genes'][0],
                    'index': true_gene_idx
                },
                'subgraph_info':{
                    'top_k': top_k,
                    'num_nodes': subgraph[0].size(0),
                    'num_edges': subgraph[1].size(1)                  
                },                
                'subgraph.triplets': {
                    'triplets': triplets                    
                }
            }
            return patient_result
        else:
            print(f"True gene {true_gene_idx} not found in subgraph with top_k={top_k}.")
            print("------------------ ||||||||||| ----------------")

    # If true gene not found in any subgraph, return None
    print(f"True gene not found in any subgraph for patient {patient_data['id']}.")
    return None

# Modified main function
def main(max_samples=None):
    print("Starting main function...")
    patients = load_patients("./converted_patients.json", max_samples)
    top_k_values = [100, 200, 400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    results = []

    for patient_data in tqdm(patients, desc="Processing patients"):
        result = process_patient(patient_data, data, node_name_to_idx, relation_dict, top_k_values)
        if result:
            results.append(result)
            # Save individual patient result
            with open(f'./patient_subgraph_data/cuda/patient_{patient_data["id"]}_result_torch.json', 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Saved result for patient {patient_data['id']} to patient_{patient_data['id']}_result.json")
        else:
            print(f"No result for patient with ID: {patient_data['id']}")

    # Save all patient results to a single JSON file
    with open('all_patients_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved all patient results to all_patients_results.json")

    # Output summary of results
    print(f"Processed {len(patients)} patients.")
    print(f"Found results for {len(results)} patients.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patients from JSON file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of patients to process. If not specified, all patients will be processed.")
    args = parser.parse_args()
    main(max_samples=args.max_samples)