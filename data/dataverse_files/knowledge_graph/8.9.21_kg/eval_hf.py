import json
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

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
    prompt += "\nTask: Based on the phenotypes, their connected genes, and the list of candidate genes, identify the true gene. Respond with ONLY the gene name (e.g., 'BSCL2', 'ETS2', etc). Do NOT include 'ANSWER:', the gene ID (e.g., 'ENSG00000168000'), or any other text. I repeat, DO NOT INCLUDE 'ANSWER:' nor gene ID (e.g., 'ENSG00000168000'). If no gene can be identified, respond with 'NONE'."
    
    return prompt

def predict_gene(prompt, model, tokenizer):
    """Predicts the true gene using the Mistral 7b model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Free memory
    del inputs, outputs
    torch.cuda.empty_cache()
    
    # Extract the gene name from the response
    predicted_gene = response.split('\n')[-1].strip()
    
    return predicted_gene

# Main Processing Function

def process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer):
    """Processes a single patient's data and predicts the true gene."""
    phenotype_mapping = map_phenotypes_to_kg(patient_data["positive_phenotypes"], hpo_to_name_dict, hpo_to_idx_dict, kg_node_map)
    all_candidate_genes_mapping = map_genes_to_kg(patient_data["all_candidate_genes"], ensembl_to_idx_dict, kg_node_map)
    phenotype_mapping = map_gene_connections_to_phenotypes(phenotype_mapping, kg_edgelist, kg_node_map, ensembl_to_idx_dict)
    
    prompt = create_prompt(phenotype_mapping, all_candidate_genes_mapping)
    predicted_gene = predict_gene(prompt, model, tokenizer)
    torch.cuda.empty_cache()  # Clear CUDA cache after prediction
    
    true_gene = patient_data["true_genes"][0]
    true_gene_name = map_genes_to_kg([true_gene], ensembl_to_idx_dict, kg_node_map)[true_gene]['node_name']
    is_correct = true_gene_name.lower() in predicted_gene.lower()
    
    result = {
        "predicted_gene": predicted_gene,
        "true_gene": true_gene_name,
        "is_correct": is_correct
    }
    return result

def process_patient_file(filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer, num_patients=None, batch_size=16):
    """Processes a file of patient data in batches and evaluates gene predictions."""
    patient_data_df = load_patient_data(filepath)
    
    if num_patients is not None:
        patient_data_df = patient_data_df.head(num_patients)
    
    results = []
    
    # Processing patients in batches
    for batch_start in range(0, len(patient_data_df), batch_size):
        batch_end = min(batch_start + batch_size, len(patient_data_df))
        batch = patient_data_df.iloc[batch_start:batch_end]
        
        batch_results = []
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Processing batch {batch_start//batch_size + 1}"):
            patient_data = json.loads(row['patient_data'])
            result = process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Free memory
        torch.cuda.empty_cache()
        
        # Print batch results
        batch_correct = sum(result['is_correct'] for result in batch_results)
        batch_accuracy = batch_correct / len(batch_results)
        print(f"\nBatch {batch_start//batch_size + 1} results:")
        print(f"Processed {len(batch_results)} patients")
        print(f"Correct predictions: {batch_correct}")
        print(f"Batch accuracy: {batch_accuracy:.2%}")
    
    correct_predictions = sum(result['is_correct'] for result in results)
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    print(f"\nOverall results:")
    print(f"Total processed patients: {total_predictions}")
    print(f"Total correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.2%}")
    
    return results, accuracy

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load necessary data
    logger.info("Loading necessary data...")
    hpo_to_name_dict = load_pickle_file('hpo_to_name_dict_8.9.21_kg.pkl')
    hpo_to_idx_dict = load_pickle_file('hpo_to_idx_dict_8.9.21_kg.pkl')
    ensembl_to_idx_dict = load_pickle_file('ensembl_to_idx_dict_8.9.21_kg.pkl')
    kg_node_map = load_csv_file('./KG_node_map_test.csv')
    kg_edgelist = load_csv_file('./KG_edgelist_mask_test.csv')

    logger.info("Loading model and tokenizer...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token=access_token, use_cache=False)
    model.gradient_checkpointing_enable()

    # Process patient file with batch processing
    filepath = '../../patients/simulated_patients/disease_split_val_short_test.txt'
    num_patients = 224
    batch_size = 16  

    logger.info(f"Starting evaluation on {num_patients} patients with batch size {batch_size}")
    results, accuracy = process_patient_file(filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer, num_patients=num_patients, batch_size=batch_size)

    logger.info(f"Final Accuracy: {accuracy:.2%}")
    logger.info("Gene prediction process completed")

    # Save results to a file
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump({"results": results, "accuracy": accuracy}, f, indent=2)
    logger.info(f"Results saved to {output_file}")