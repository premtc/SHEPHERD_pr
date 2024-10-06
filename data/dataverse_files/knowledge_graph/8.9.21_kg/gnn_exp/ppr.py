import numpy as np
import pandas as pd
import networkx as nx
import json
import pickle
import os

# Load data or graph
try:
    print("Trying to load the knowledge graph from file...")
    with open('knowledge_graph_ppr.pkl', 'rb') as f:
        G = pickle.load(f)
    print("Knowledge graph loaded successfully.")

    # Load data for mapping node names to indices
    print("Loading node and edge data for node mapping...")
    nodes_df = pd.read_csv('../KG_node_map_test.csv')
    edges_df = pd.read_csv('../KG_edgelist_mask_test.csv')
except FileNotFoundError:
    print("Knowledge graph file not found. Creating the knowledge graph...")
    # Load data
    print("Loading node and edge data...")
    nodes_df = pd.read_csv('../KG_node_map_test.csv')
    edges_df = pd.read_csv('../KG_edgelist_mask_test.csv')

    # Create the knowledge graph
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        node_idx = row['node_idx']
        G.add_node(node_idx, node_id=node_idx, node_type=row['node_type'], node_name=row['node_name'])
        print(f"Added node: {node_idx}")

    for _, row in edges_df.iterrows():
        source = row['x_idx']
        target = row['y_idx']
        relation = row['full_relation']
        G.add_edge(source, target, relation=relation)
        print(f"Added edge from {source} to {target} with relation: {relation}")

    # Save the knowledge graph for future use
    with open('knowledge_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Knowledge graph saved to file.")

# Map node names to indices
print("Mapping node names to indices...")
node_name_to_idx = {row['node_name']: row['node_idx'] for _, row in nodes_df.iterrows()}

# Create a dictionary to map relation types to numeric values
print("Creating relation type dictionary...")
relation_types = edges_df['full_relation'].unique()
relation_dict = {relation: idx for idx, relation in enumerate(relation_types)}
print(f"Relation dictionary length: {len(relation_dict)}")

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

# Function to compute personalized PageRank
def compute_ppr(G, personalization_nodes, alpha=0.15):
    print(f"Computing personalized PageRank for nodes: {personalization_nodes} with alpha={alpha}...")
    personalization = {node: 0 for node in G.nodes()}
    for node in personalization_nodes:
        personalization[node] = 1 / len(personalization_nodes)
    ppr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization)
    return ppr_scores

# Function to extract subgraph based on PPR scores
def extract_subgraph(ppr_scores, G, top_k):
    print(f"Extracting subgraph with top {top_k} nodes...")
    top_nodes = sorted(ppr_scores, key=ppr_scores.get, reverse=True)[:top_k]
    subgraph = G.subgraph(top_nodes).copy()
    print(f"Extracted subgraph nodes: {list(subgraph.nodes())}")
    return subgraph

# Function to extract triplets from subgraph
def extract_triplets_from_subgraph(subgraph, relation_dict):
    print("Extracting triplets from subgraph...")
    triplets = []
    for source, target, data in subgraph.edges(data=True):
        relation = data.get('relation', 'unknown')
        relation_numeric = relation_dict.get(relation, -1)
        triplets.append((source, relation_numeric, target))
    print(f"Extracted triplets length: {len(triplets)}")
    return triplets

# Function to process each patient
def process_patient(patient_data, G, node_name_to_idx, relation_dict, top_k_values):
    print(f"Processing patient with ID: {patient_data['id']}...")
    phenotype_indices = map_phenotypes_to_indices(patient_data['positive_phenotypes'], node_name_to_idx)
    true_gene_idx = map_gene_to_index(patient_data['true_genes'][0], node_name_to_idx)

    if true_gene_idx is None:
        print(f"True gene not found in KG for patient {patient_data['id']}")
        return None

    for top_k in top_k_values:
        print(f"Trying top_k={top_k}...")
        # Compute PPR scores
        ppr_scores = compute_ppr(G, phenotype_indices)
        
        # Extract subgraph
        subgraph = extract_subgraph(ppr_scores, G, top_k=top_k)
        
        # Check if true gene is in subgraph
        if true_gene_idx in subgraph.nodes():
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
                    'num_nodes': len(subgraph.nodes()),
                    'num_edges': len(subgraph.edges())                  
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

# Example usage with multiple patients
def main():
    print("Starting main function...")
    patient_samples = [
        {
            'id': 1,
            'positive_phenotypes': ['Hypogonadism', 'Increased circulating ferritin concentration', 'Generalized hyperpigmentation', 'Abnormality of iron homeostasis', 'Elevated transferrin saturation', 'Hepatic fibrosis', 'Abnormal liver morphology', 'Abnormality of pancreas physiology', 'Abnormality of endocrine pancreas physiology', 'Overbite', 'Intestinal bleeding','Vitreomacular adhesion', 'Gangrene', 'Abnormality of the outer ear', 'Lissencephaly', 'Oppositional defiant disorder'],
            'true_genes': ['HAMP']
        },
        {
            'id': 2,
            'positive_phenotypes': ['Elevated hemoglobin A1c', 'Muscular hypotonia of the trunk', 'Mild global developmental delay', 'Downturned corners of mouth', 'Abnormal blood glucose concentration', 'Short nose', 'Cardiomyopathy', 'Glucose intolerance', 'Diarrhea', 'Hyperkinetic movements', 'Hyperactivity', 'Menorrhagia',  'Aplasia of the nose', 'Chronic myelogenous leukemia', 'Proteinuria', 'Microphakia', 'Laryngeal edema'],
            'true_genes': ['ABCC8']
        }
    ]
    top_k_values = [100, 200, 400, 800, 1000, 1500, 2000, 2500]
    results = []

    for patient_data in patient_samples:
        print(f"Processing patient sample with ID: {patient_data['id']}...")
        result = process_patient(patient_data, G, node_name_to_idx, relation_dict, top_k_values)
        if result:
            results.append(result)
        else:
            print(f"No result for patient with ID: {patient_data['id']}")

    # Save all patient results to a single JSON file
    with open('all_patients_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved all patient results to all_patients_results.json")

    # Output the results
    print("Final results:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()