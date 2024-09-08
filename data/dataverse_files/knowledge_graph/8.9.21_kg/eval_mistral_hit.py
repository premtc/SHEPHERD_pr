import json
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import gc
import csv
import logging
from datetime import datetime


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

def create_prompt(phenotype_mapping, candidate_genes):
    """Creates a strict prompt for the LLM to rank candidate genes."""
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
    prompt += "\nTASK: Rank ALL candidate genes from most to least likely to be the causative gene based on the given phenotypes and gene connections. You MUST follow these rules:\n"
    prompt += "1. Output ONLY a comma-separated list of gene names (not gene IDs).\n"
    prompt += "2. Include ALL candidate genes in your ranking.\n"
    prompt += "3. Do NOT explain your reasoning or include any other text.\n"
    prompt += "4. If you're unsure, make an educated guess. Do NOT skip any genes.\n"
    prompt += "5. Use ONLY the gene names (e.g., SCARB2, not ENSG00000123570:SCARB2).\n"
    prompt += "6. Your output should look EXACTLY like this example:\n"
    prompt += "SCARB2,SPTA1,DISP1,IDH1,KIAA1549,PEX19,FOXN1,MYD88,PTPN3,ADAMTS13\n"
    prompt += "7. Do NOT include words like 'Ranking:' or 'Predicted Ranking:' in your output.\n"
    prompt += "8. If you absolutely cannot produce a ranking, output only the word 'NONE'.\n\n"
    prompt += "Now, provide your ranking:"
    
    return prompt

def predict_gene_ranking(prompt, model, tokenizer):
    """Predicts the ranking of candidate genes using the Mistral 7b model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, num_return_sequences=1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Free memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    # Extract the gene ranking from the response
    predicted_ranking = response.split('\n')[-1].strip()
    
    return predicted_ranking

def parse_gene_ranking(ranking_string):
    """Parses the comma-separated string of gene names into a list."""
    if ranking_string.upper() == 'NONE':
        return []
    
    # Remove any potential leading/trailing whitespace and split by comma
    genes = [gene.strip() for gene in ranking_string.split(',') if gene.strip()]
    
    # Remove any genes that contain ':' (likely gene IDs)
    genes = [gene for gene in genes if ':' not in gene]
    
    return genes

def evaluate_ranking(predicted_ranking, true_gene):
    """Evaluates the ranking based on hit@1, hit@5, and hit@10."""
    if true_gene in predicted_ranking:
        rank = predicted_ranking.index(true_gene) + 1
        hit_1 = 1 if rank == 1 else 0
        hit_5 = 1 if rank <= 5 else 0
        hit_10 = 1 if rank <= 10 else 0
    else:
        hit_1, hit_5, hit_10 = 0, 0, 0
    
    return hit_1, hit_5, hit_10

def process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer):
    """Processes a single patient's data and predicts the ranking of candidate genes."""
    try:
        # Check if the number of positive phenotypes is too high
        if len(patient_data["positive_phenotypes"]) > MAX_SAMPLE_LENGTH:
            return {
                "predicted_ranking": [],
                "true_gene": "SKIPPED",
                "hit@1": 0,
                "hit@5": 0,
                "hit@10": 0,
                "reason": "Too many phenotypes"
            }

        phenotype_mapping = map_phenotypes_to_kg(patient_data["positive_phenotypes"], hpo_to_name_dict, hpo_to_idx_dict, kg_node_map)
        all_candidate_genes_mapping = map_genes_to_kg(patient_data["all_candidate_genes"], ensembl_to_idx_dict, kg_node_map)
        phenotype_mapping = map_gene_connections_to_phenotypes(phenotype_mapping, kg_edgelist, kg_node_map, ensembl_to_idx_dict)
        
        prompt = create_prompt(phenotype_mapping, all_candidate_genes_mapping)
        predicted_ranking_string = predict_gene_ranking(prompt, model, tokenizer)
        predicted_ranking = parse_gene_ranking(predicted_ranking_string)
        
        true_gene = patient_data["true_genes"][0]
        true_gene_name = map_genes_to_kg([true_gene], ensembl_to_idx_dict, kg_node_map)[true_gene]['node_name']
        hit_1, hit_5, hit_10 = evaluate_ranking(predicted_ranking, true_gene_name)
        
        result = {
            "predicted_ranking": predicted_ranking,
            "true_gene": true_gene_name,
            "hit@1": hit_1,
            "hit@5": hit_5,
            "hit@10": hit_10,
            "reason": "Processed successfully"
        }
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA out of memory error
        torch.cuda.empty_cache()  # Clear CUDA memory
        result = {
            "predicted_ranking": [],
            "true_gene": "SKIPPED",
            "hit@1": 0,
            "hit@5": 0,
            "hit@10": 0,
            "reason": "CUDA out of memory"
        }
    except Exception as e:
        # Handle any other unexpected errors
        result = {
            "predicted_ranking": [],
            "true_gene": "SKIPPED",
            "hit@1": 0,
            "hit@5": 0,
            "hit@10": 0,
            "reason": f"Error: {str(e)}"
        }
    
    return result

def process_patient_file(filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer, num_patients=None, batch_size=4):
    """Processes a file of patient data in batches and evaluates gene rankings."""
    patient_data_df = load_patient_data(filepath)
    
    if num_patients is not None:
        patient_data_df = patient_data_df.head(num_patients)
    
    total_hit_1 = 0
    total_hit_5 = 0
    total_hit_10 = 0
    total_processed = 0
    total_skipped_samples = 0
    total_cuda_oom_samples = 0
    total_other_error_samples = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"results_summary_{timestamp}.csv")
    
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Predicted_Ranking', 'Ground_Truth', 'Hit@1', 'Hit@5', 'Hit@10'])
    
    # Processing patients in batches
    for batch_start in range(0, len(patient_data_df), batch_size):
        batch_end = min(batch_start + batch_size, len(patient_data_df))
        batch = patient_data_df.iloc[batch_start:batch_end]
        
        batch_hit_1 = 0
        batch_hit_5 = 0
        batch_hit_10 = 0
        batch_processed = 0
        batch_skipped_samples = 0
        batch_cuda_oom_samples = 0
        batch_other_error_samples = 0
        
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Processing batch {batch_start//batch_size + 1}"):
            patient_data = json.loads(row['patient_data'])
            result = process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer)
            
            if result["true_gene"] == "SKIPPED":
                if result["reason"] == "CUDA out of memory":
                    batch_cuda_oom_samples += 1
                elif result["reason"] == "Too many phenotypes":
                    batch_skipped_samples += 1
                else:
                    batch_other_error_samples += 1
            else:
                batch_processed += 1
                batch_hit_1 += result['hit@1']
                batch_hit_5 += result['hit@5']
                batch_hit_10 += result['hit@10']
                
                # Write result to CSV
                with open(csv_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([', '.join(result['predicted_ranking']), result['true_gene'], result['hit@1'], result['hit@5'], result['hit@10']])
                
                print(f"\nPredicted Ranking: {result['predicted_ranking'][:10]}...")
                print(f"Ground Truth: {result['true_gene']}")
                print(f"Hit@1: {result['hit@1']}, Hit@5: {result['hit@5']}, Hit@10: {result['hit@10']}")
        
        # Update total counts
        total_hit_1 += batch_hit_1
        total_hit_5 += batch_hit_5
        total_hit_10 += batch_hit_10
        total_processed += batch_processed
        total_skipped_samples += batch_skipped_samples
        total_cuda_oom_samples += batch_cuda_oom_samples
        total_other_error_samples += batch_other_error_samples

        # Clear CUDA memory after each batch to prevent memory overflow
        torch.cuda.empty_cache()
        gc.collect()

        # Add a check before calculating and printing accuracy
        if batch_processed > 0:
            batch_hit_1_accuracy = batch_hit_1 / batch_processed
            batch_hit_5_accuracy = batch_hit_5 / batch_processed
            batch_hit_10_accuracy = batch_hit_10 / batch_processed
        else:
            batch_hit_1_accuracy = 0
            batch_hit_5_accuracy = 0
            batch_hit_10_accuracy = 0

        # Print batch results
        print(f"\nBatch {batch_start//batch_size + 1} results:")
        print(f"Processed {batch_processed} patients")
        print(f"Batch Hit@1: {batch_hit_1_accuracy:.2%}")
        print(f"Batch Hit@5: {batch_hit_5_accuracy:.2%}")
        print(f"Batch Hit@10: {batch_hit_10_accuracy:.2%}")
        print(f"Batch skipped samples: {batch_skipped_samples}")
        print(f"Batch CUDA OOM samples: {batch_cuda_oom_samples}")
        print(f"Batch other error samples: {batch_other_error_samples}")
        print(f"Total processed so far: {total_processed}")
        print(f"Total Hit@1 so far: {total_hit_1 / total_processed:.2%}")
        print(f"Total Hit@5 so far: {total_hit_5 / total_processed:.2%}")
        print(f"Total Hit@10 so far: {total_hit_10 / total_processed:.2%}")
    
    hit_1_accuracy = total_hit_1 / total_processed if total_processed > 0 else 0
    hit_5_accuracy = total_hit_5 / total_processed if total_processed > 0 else 0
    hit_10_accuracy = total_hit_10 / total_processed if total_processed > 0 else 0
    
    print(f"\nOverall results:")
    print(f"Total processed patients: {total_processed}")
    print(f"Hit@1 Accuracy: {hit_1_accuracy:.2%}")
    print(f"Hit@5 Accuracy: {hit_5_accuracy:.2%}")
    print(f"Hit@10 Accuracy: {hit_10_accuracy:.2%}")
    print(f"Total skipped samples: {total_skipped_samples}")
    print(f"Total CUDA OOM samples: {total_cuda_oom_samples}")
    print(f"Total other error samples: {total_other_error_samples}")
    
    return hit_1_accuracy, hit_5_accuracy, hit_10_accuracy, total_skipped_samples, total_cuda_oom_samples, total_other_error_samples

if __name__ == "__main__":
    import logging
    from transformers import BitsAndBytesConfig
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Loading necessary data...")
    hpo_to_name_dict = load_pickle_file('hpo_to_name_dict_8.9.21_kg.pkl')
    hpo_to_idx_dict = load_pickle_file('hpo_to_idx_dict_8.9.21_kg.pkl')
    ensembl_to_idx_dict = load_pickle_file('ensembl_to_idx_dict_8.9.21_kg.pkl')
    kg_node_map = load_csv_file('./KG_node_map_test.csv')
    kg_edgelist = load_csv_file('./KG_edgelist_mask_test.csv')
    MAX_SAMPLE_LENGTH = 720

    logger.info("Loading model and tokenizer...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        token=access_token,
        use_cache=False
    )
    model.gradient_checkpointing_enable()

    filepath = '../../patients/simulated_patients/disease_split_val_short_test.txt'
    num_patients = 6500
    batch_size = 1  


    logger.info(f"Starting evaluation on {num_patients} patients with batch size {batch_size}")
    hit_1, hit_5, hit_10, skipped_samples, cuda_oom_samples, other_error_samples = process_patient_file(
        filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, 
        model, tokenizer, num_patients=num_patients, batch_size=batch_size
    )

    logger.info(f"Hit@1 Accuracy: {hit_1:.2%}")
    logger.info(f"Hit@5 Accuracy: {hit_5:.2%}")
    logger.info(f"Hit@10 Accuracy: {hit_10:.2%}")
    logger.info(f"Total skipped samples: {skipped_samples}")
    logger.info(f"Total CUDA OOM samples: {cuda_oom_samples}")
    logger.info(f"Total other error samples: {other_error_samples}")
    logger.info("Gene ranking evaluation completed")

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "hit@1": hit_1,
            "hit@5": hit_5,
            "hit@10": hit_10,
            "skipped_samples": skipped_samples,
            "cuda_oom_samples": cuda_oom_samples,
            "other_error_samples": other_error_samples
        }, f, indent=2)
    logger.info(f"Results saved to {output_file}")