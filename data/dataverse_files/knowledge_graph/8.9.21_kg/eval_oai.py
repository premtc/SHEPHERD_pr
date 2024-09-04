import json
import pandas as pd
import pickle
from openai import OpenAI
from tqdm import tqdm

# Utility Functions

def load_patient_data(filepath):
    """Loads patient data from a text file."""
    return pd.read_csv(filepath, sep='\t', header=None, names=["patient_data"])

def load_pickle_file(filepath):
    """Loads a pickle file and returns the stored dictionary."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_csv_file(filepath):
    """Loads a CSV file into a DataFrame."""
    return pd.read_csv(filepath)

# Mapping Functions

def map_phenotypes_to_kg(phenotypes, hpo_to_name_dict, hpo_to_idx_dict, kg_node_map):
    """Maps phenotypes to the KG nodes based on the node map."""
    phenotype_mapping = {}
    
    for hpo in phenotypes:
        phenotype_name = hpo_to_name_dict.get(hpo, "Unknown HPO Code")
        phenotype_idx = hpo_to_idx_dict.get(hpo, "Unknown HPO Code")
        
        if phenotype_idx != "Unknown HPO Code":
            matching_row = kg_node_map[kg_node_map['node_idx'] == phenotype_idx]
            
            if not matching_row.empty:
                phenotype_mapping[hpo] = {
                    'name': phenotype_name,
                    'node_idx': phenotype_idx,
                    'node_id': matching_row['node_id'].values[0],
                    'node_type': matching_row['node_type'].values[0]
                }
            else:
                phenotype_mapping[hpo] = {
                    'name': phenotype_name,
                    'node_idx': phenotype_idx,
                    'node_id': "Not in KG",
                    'node_type': "Not in KG"
                }
        else:
            phenotype_mapping[hpo] = {
                'name': phenotype_name,
                'node_idx': "Unknown HPO Code",
                'node_id': "Not in KG",
                'node_type': "Not in KG"
            }
    
    return phenotype_mapping

def map_genes_to_kg(genes, ensembl_to_idx_dict, kg_node_map):
    """Maps Ensembl gene IDs to KG nodes."""
    gene_mapping = {}
    
    for gene in genes:
        gene_idx = ensembl_to_idx_dict.get(gene, "Unknown Ensembl ID")
        matching_row = kg_node_map[kg_node_map['node_idx'] == gene_idx]
        
        if not matching_row.empty:
            gene_mapping[gene] = {
                'node_idx': gene_idx,
                'node_id': matching_row['node_id'].values[0],
                'node_type': matching_row['node_type'].values[0],
                'node_name': matching_row['node_name'].values[0]
            }
        else:
            gene_mapping[gene] = {
                'node_idx': gene_idx,
                'node_id': "Not in KG",
                'node_type': "Not in KG",
                'node_name': "Not in KG"
            }
    
    return gene_mapping

def map_gene_connections_to_phenotypes(phenotype_mapping, kg_edgelist, kg_node_map, ensembl_to_idx_dict):
    """Maps gene connections to each phenotype based on the KG edgelist."""
    idx_to_ensembl = {v: k for k, v in ensembl_to_idx_dict.items()}
    
    for hpo, info in phenotype_mapping.items():
        phenotype_idx = info['node_idx']
        
        if phenotype_idx != "Unknown HPO Code" and info['node_id'] != "Not in KG":
            connected_genes = kg_edgelist[
                (kg_edgelist['x_idx'] == phenotype_idx) &
                (kg_edgelist['full_relation'].str.contains("effect/phenotype;phenotype_protein;gene/protein"))
            ]
            
            gene_indices = connected_genes['y_idx'].tolist()
            gene_info = kg_node_map[kg_node_map['node_idx'].isin(gene_indices)][['node_idx', 'node_name']]
            
            phenotype_mapping[hpo]['connected_genes'] = [
                f"{idx_to_ensembl.get(idx, 'Unknown')}:{name}"
                for idx, name in gene_info.itertuples(index=False, name=None)
            ]
        else:
            phenotype_mapping[hpo]['connected_genes'] = []
    
    return phenotype_mapping

# LLM Interaction Functions

def create_prompt(phenotype_mapping, candidate_genes):
    """Creates a prompt for the LLM based on phenotype and gene data."""
    prompt = "Patient Data:\n"
    
    # Phenotypes and their connected genes
    prompt += "Phenotypes and Connected Genes:\n"
    for hpo, mapping in phenotype_mapping.items():
        connected_genes = mapping['connected_genes']
        gene_list = ", ".join(connected_genes)
        prompt += f"- {hpo} - {mapping['name']}\n"
        prompt += f"  Connected Genes: {gene_list if gene_list else 'None'}\n"
    
    # Candidate genes
    candidate_gene_list = ", ".join([f"{gene}:{info['node_name']}" for gene, info in candidate_genes.items()])
    prompt += f"\nCandidate Genes:\n{candidate_gene_list}\n"
    
    # Task description
    prompt += "\nTask: Based on the phenotypes, their connected genes, and the list of candidate genes, identify the true gene. Please respond with only the name of the true gene, nothing else."
    
    return prompt

def predict_gene(prompt, api_key):
    """Predicts the true gene using the OpenAI API."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-4o-mini" if that's the model you're using
        messages=[
            {"role": "system", "content": "You are a genetic analysis assistant. Your task is to predict the true gene based on the given phenotypes and candidate genes."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Main Processing Function

def process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, api_key):
    """Processes a single patient's data and predicts the true gene."""
    # Step 1: Map phenotypes to the KG
    phenotype_mapping = map_phenotypes_to_kg(patient_data["positive_phenotypes"], hpo_to_name_dict, hpo_to_idx_dict, kg_node_map)
    
    # Step 2: Map all candidate genes to the KG
    all_candidate_genes_mapping = map_genes_to_kg(patient_data["all_candidate_genes"], ensembl_to_idx_dict, kg_node_map)
    
    # Step 3: Map gene connections to phenotypes
    phenotype_mapping = map_gene_connections_to_phenotypes(phenotype_mapping, kg_edgelist, kg_node_map, ensembl_to_idx_dict)
    
    # Step 4: Create prompt for LLM
    prompt = create_prompt(phenotype_mapping, all_candidate_genes_mapping)
    
    # Step 5: Get prediction from LLM
    predicted_gene = predict_gene(prompt, api_key)
    
    # Step 6: Get true gene
    true_gene = patient_data["true_genes"][0]
    true_gene_name = map_genes_to_kg([true_gene], ensembl_to_idx_dict, kg_node_map)[true_gene]['node_name']
    
    # Step 7: Compare prediction to true gene
    is_correct = predicted_gene.lower() == true_gene_name.lower()
    
    return {
        "predicted_gene": predicted_gene,
        "true_gene": true_gene_name,
        "is_correct": is_correct
    }

# File Processing Function

def process_patient_file(filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, api_key, num_patients=None):
    """Processes a file of patient data and evaluates gene predictions."""
    patient_data_df = load_patient_data(filepath)
    
    if num_patients is not None:
        patient_data_df = patient_data_df.head(num_patients)
    
    results = []
    
    for _, row in tqdm(patient_data_df.iterrows(), total=len(patient_data_df), desc="Processing patients"):
        patient_data = json.loads(row['patient_data'])
        result = process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, api_key)
        results.append(result)
    
    correct_predictions = sum(result['is_correct'] for result in results)
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    print(f"Processed {total_predictions} patients")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return results, accuracy

# Main Execution

if __name__ == "__main__":
    # Load necessary data
    hpo_to_name_dict = load_pickle_file('hpo_to_name_dict_8.9.21_kg.pkl')
    hpo_to_idx_dict = load_pickle_file('hpo_to_idx_dict_8.9.21_kg.pkl')
    ensembl_to_idx_dict = load_pickle_file('ensembl_to_idx_dict_8.9.21_kg.pkl')
    kg_node_map = load_csv_file('./KG_node_map_test.csv')
    kg_edgelist = load_csv_file('./KG_edgelist_mask_test.csv')

    # Set up OpenAI API
    
    

    # Process patient file
    filepath = '../../patients/simulated_patients/disease_split_val_short_test.txt'
    results, accuracy = process_patient_file(filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, api_key, num_patients=10)

    # Print final results
    print(f"Final Accuracy: {accuracy:.2%}")