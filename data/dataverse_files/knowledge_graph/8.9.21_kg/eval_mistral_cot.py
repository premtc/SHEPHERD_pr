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
import wandb  # Import WandB

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
    prompt = "Based on the following patient data, identify the true gene from the list of candidate genes.\n\nPatient Phenotypes:\n"
    
    # Add phenotypes with names
    for hpo, mapping in phenotype_mapping.items():
        prompt += f"- {mapping['name']} ({hpo})\n"
    
    # Candidate genes
    prompt += "\nCandidate Genes:\n"
    for gene, info in candidate_genes.items():
        prompt += f"- {info['node_name']} ({gene})\n"
    
    # Task description with CoT
    prompt += (
        "\nPlease analyze the patient phenotypes and determine which candidate gene is most likely associated."
        "\nProvide a step-by-step reasoning process, and then give your final answer."
        "\nYour final answer should start with 'Answer:' and only include the gene name from the candidate genes."
        "\n\nReasoning:\n"
    )

    print("=== Prompt ===")
    print(prompt)
    print("==============")
    
    return prompt
# def create_prompt(phenotype_mapping, candidate_genes):
#     """Creates a simplified and clear prompt for the LLM."""
#     prompt = "Patient Phenotypes:\n"
#     for hpo, mapping in phenotype_mapping.items():
#         prompt += f"- {mapping['name']}\n"
    
#     prompt += "\nCandidate Genes:\n"
#     for gene, info in candidate_genes.items():
#         prompt += f"- {info['node_name']}\n"
    
#     prompt += (
#         "\nQuestion:\n"
#         "Based on the patient phenotypes listed above, which candidate gene is most likely associated with the patient?\n"
#         "Please provide your final answer in the format 'Answer: [Gene Name]'.\n\n"
#         "Answer:"
#     )
#     return prompt

def predict_gene(prompt, model, tokenizer, candidate_gene_names):
    """Predicts the true gene using the LLM with Chain-of-Thought."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_return_sequences=1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Free memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    # Extract the generated text after the prompt
    generated_text = response[len(prompt):].strip()
    
    # Initialize variables
    reasoning_text = generated_text
    predicted_gene = "NONE"
    
    # Split the reasoning and the answer
    if "Answer:" in generated_text:
        reasoning_text, answer_text = generated_text.split("Answer:", 1)
        reasoning_text = reasoning_text.strip()
        answer_text = answer_text.strip().split('\n')[0]  # Get the first line after 'Answer:'
        predicted_gene = answer_text.strip()
    else:
        # If 'Answer:' is not found, attempt to extract the last candidate gene mentioned
        # Search for candidate gene names in the reasoning text
        for gene_name in reversed(candidate_gene_names):
            if gene_name.lower() in reasoning_text.lower():
                predicted_gene = gene_name
                break
    
    # Ensure the predicted gene is one of the candidate genes
    if predicted_gene not in candidate_gene_names:
        predicted_gene = "NONE"
    
    return predicted_gene, reasoning_text, response  # Return the full response for logging

def process_single_patient(patient_data, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer):
    """Processes a single patient's data and predicts the true gene."""
    try:
        # Check if the number of positive phenotypes is too high
        if len(patient_data["positive_phenotypes"]) > MAX_SAMPLE_LENGTH:
            return {
                "predicted_gene": "NONE",
                "true_gene": "SKIPPED",
                "is_correct": False,
                "reason": "Too many phenotypes",
                "chain_of_thought": ""
            }

        phenotype_mapping = map_phenotypes_to_kg(patient_data["positive_phenotypes"], hpo_to_name_dict, hpo_to_idx_dict, kg_node_map)
        all_candidate_genes_mapping = map_genes_to_kg(patient_data["all_candidate_genes"], ensembl_to_idx_dict, kg_node_map)
        phenotype_mapping = map_gene_connections_to_phenotypes(phenotype_mapping, kg_edgelist, kg_node_map, ensembl_to_idx_dict)
        
        # Get candidate gene names
        candidate_gene_names = [info['node_name'] for gene, info in all_candidate_genes_mapping.items()]
        
        prompt = create_prompt(phenotype_mapping, all_candidate_genes_mapping)
        predicted_gene, reasoning, full_response = predict_gene(prompt, model, tokenizer, candidate_gene_names)
        
        # Map true gene Ensembl ID to gene name
        true_gene = patient_data["true_genes"][0]
        true_gene_name = map_genes_to_kg([true_gene], ensembl_to_idx_dict, kg_node_map)[true_gene]['node_name']
        is_correct = predicted_gene.lower() == true_gene_name.lower()
        
        result = {
            "predicted_gene": predicted_gene,
            "true_gene": true_gene_name,
            "is_correct": is_correct,
            "reason": "Processed successfully",
            "chain_of_thought": reasoning,
            "full_response": full_response
        }
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA out of memory error
        torch.cuda.empty_cache()  # Clear CUDA memory
        result = {
            "predicted_gene": "NONE",
            "true_gene": "SKIPPED",
            "is_correct": False,
            "reason": "CUDA out of memory",
            "chain_of_thought": "",
            "full_response": ""
        }
    except Exception as e:
        # Handle any other unexpected errors
        result = {
            "predicted_gene": "NONE",
            "true_gene": "SKIPPED",
            "is_correct": False,
            "reason": f"Error: {str(e)}",
            "chain_of_thought": "",
            "full_response": ""
        }
    
    return result

def process_patient_file(filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, model, tokenizer, num_patients=None, batch_size=4):
    """Processes a file of patient data in batches and evaluates gene predictions."""
    patient_data_df = load_patient_data(filepath)
    
    if num_patients is not None:
        patient_data_df = patient_data_df.head(num_patients)
    
    total_correct = 0
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
        csvwriter.writerow(['Predicted', 'Ground_Truth', 'Correct', 'Reason', 'Chain_of_Thought'])
    
    # Processing patients in batches
    for batch_start in range(0, len(patient_data_df), batch_size):
        batch_end = min(batch_start + batch_size, len(patient_data_df))
        batch = patient_data_df.iloc[batch_start:batch_end]
        
        batch_correct = 0
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
                if result['is_correct']:
                    batch_correct += 1
                
                # Write result to CSV
                with open(csv_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([
                        result['predicted_gene'], 
                        result['true_gene'], 
                        result['is_correct'], 
                        result['reason'],
                        result['chain_of_thought']
                    ])
                
                print(f"\nPredicted: {result['predicted_gene']}, Ground Truth: {result['true_gene']}, Correct: {result['is_correct']}")
                # Print chain of thought for debugging
                print(f"Chain of Thought:\n{result['chain_of_thought']}\n")
            
            # Log to WandB
            wandb.log({
                'batch': batch_start // batch_size + 1,
                'predicted_gene': result['predicted_gene'],
                'true_gene': result['true_gene'],
                'is_correct': result['is_correct'],
                'reason': result['reason'],
                'chain_of_thought': result['chain_of_thought'],
                'full_response': result['full_response']
            })
        
        # Update total counts
        total_correct += batch_correct
        total_processed += batch_processed
        total_skipped_samples += batch_skipped_samples
        total_cuda_oom_samples += batch_cuda_oom_samples
        total_other_error_samples += batch_other_error_samples
        
        # Clear CUDA memory after each batch to prevent memory overflow
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print batch results
        batch_accuracy = batch_correct / batch_processed if batch_processed > 0 else 0
        print(f"\nBatch {batch_start//batch_size + 1} results:")
        print(f"Processed {batch_processed} patients")
        print(f"Correct predictions: {batch_correct}")
        print(f"Batch accuracy: {batch_accuracy:.2%}")
        print(f"Batch skipped samples: {batch_skipped_samples}")
        print(f"Batch CUDA OOM samples: {batch_cuda_oom_samples}")
        print(f"Batch other error samples: {batch_other_error_samples}")
        print(f"Total processed so far: {total_processed}")
        print(f"Total correct so far: {total_correct}")
        print(f"Total skipped samples so far: {total_skipped_samples}")
        print(f"Total CUDA OOM samples so far: {total_cuda_oom_samples}")
        print(f"Total other error samples so far: {total_other_error_samples}")
    
    accuracy = total_correct / total_processed if total_processed > 0 else 0
    
    print(f"\nOverall results:")
    print(f"Total processed patients: {total_processed}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Total skipped samples: {total_skipped_samples}")
    print(f"Total CUDA OOM samples: {total_cuda_oom_samples}")
    print(f"Total other error samples: {total_other_error_samples}")
    
    # Log final results to WandB
    wandb.log({
        'total_processed': total_processed,
        'total_correct': total_correct,
        'accuracy': accuracy,
        'total_skipped_samples': total_skipped_samples,
        'total_cuda_oom_samples': total_cuda_oom_samples,
        'total_other_error_samples': total_other_error_samples
    })
    
    return accuracy, total_skipped_samples, total_cuda_oom_samples, total_other_error_samples

if __name__ == "__main__":
    import logging
    from transformers import BitsAndBytesConfig
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize WandB
    wandb.init(project='gene_prediction_with_CoT', name='gene_prediction_run')

    logger.info("Loading necessary data...")
    hpo_to_name_dict = load_pickle_file('hpo_to_name_dict_8.9.21_kg.pkl')
    hpo_to_idx_dict = load_pickle_file('hpo_to_idx_dict_8.9.21_kg.pkl')
    ensembl_to_idx_dict = load_pickle_file('ensembl_to_idx_dict_8.9.21_kg.pkl')
    kg_node_map = load_csv_file('./KG_node_map_test.csv')
    kg_edgelist = load_csv_file('./KG_edgelist_mask_test.csv')
    MAX_SAMPLE_LENGTH = 720

    logger.info("Loading model and tokenizer...")
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use an instruction-following model

    

    # Configure quantization (optional)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            token=access_token,
            torch_dtype=torch.float16
        )
        model.gradient_checkpointing_enable()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    filepath = '../../patients/simulated_patients/disease_split_val_short_test.txt'
    num_patients = 1200
    batch_size = 1  

    logger.info(f"Starting evaluation on {num_patients} patients with batch size {batch_size}")
    results = process_patient_file(
        filepath, hpo_to_name_dict, hpo_to_idx_dict, ensembl_to_idx_dict, kg_node_map, kg_edgelist, 
        model, tokenizer, num_patients=num_patients, batch_size=batch_size
    )

    logger.info("Gene prediction process completed")

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "results": results
        }, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Finish WandB run
    wandb.finish()