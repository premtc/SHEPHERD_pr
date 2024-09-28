import pandas as pd

def get_node_id(node_name, df_nm):
    """Returns the node_id for a given node name"""
    return df_nm[df_nm['node_name'] == node_name]['node_idx'].values[0]

def get_connections(node_id, df_el):
    """Returns the connections of a node id"""
    return df_el[df_el['x_idx'] == node_id]['y_idx'].tolist()

def get_multi_hop_connections(start_node, max_hops, df_nm, df_el):
    """Returns connections up to max_hops for a given start node"""
    start_id = get_node_id(start_node, df_nm)

    # Initialize connections
    # connections[i] will store the connections at i hops
    # For each hop, we store the connections in a set to avoid duplicates
    connections = [set() for _ in range(max_hops)]
    # output is going to be [{}, {}, {}]

    # Initialize current_hop with the start node
    current_hop = {start_id}
    
    for i in range(max_hops):
        next_hop = set()
        for node in current_hop:
            new_connections = set(get_connections(node, df_el)) - set.union(*connections[:i+1], {start_id})
            connections[i].update(new_connections)
            next_hop.update(new_connections)
        current_hop = next_hop
        if not current_hop:
            break
    print(type(connections))
    print(len(connections))
    print(connections)
    return connections

def find_connection_hop(start_node, end_node, df_nm, df_el, max_hops=2):
    """Finds the hop distance between start_node and end_node"""
    end_id = get_node_id(end_node, df_nm)
    connections = get_multi_hop_connections(start_node, max_hops, df_nm, df_el)
    
    for i, hop in enumerate(connections, 1):
        if end_id in hop:
            return f"{i}-hop connection"
    return f"Not found within {max_hops} hops"

def check_connections_for_node_list(start_node_list, end_node, df_nm, df_el, max_hops=2):
    """Checks connections between a list of start nodes and an end node"""
    results = []
    for start_node in start_node_list:
        result = find_connection_hop(start_node, end_node, df_nm, df_el, max_hops)
        results.append((start_node, result))
    return results

# Example usage
def main(phenotype_list, gene_name, df_nm, df_el):
    results = check_connections_for_node_list(phenotype_list, gene_name, df_nm, df_el)
    for phenotype, result in results:
        print(f"Phenotype: {phenotype}")
        print(f"Result: {result}")
        print()

# If you want to run this script directly
if __name__ == "__main__":
    # Load your dataframes here if needed
    df_nm = pd.read_csv('./KG_node_map_test.csv')
    df_el = pd.read_csv('./KG_edgelist_mask_test.csv')
    
    # phenotype_list = ['Abnormal acetabulum morphology', 'Insomnia', 'Hyperactivity', 'Hyperkinetic movements', 'Abnormal posturing', 'Dental crowding', 'Overjet']
    phenotype_list = ['Elevated hemoglobin A1c', 'Muscular hypotonia of the trunk', 'Mild global developmental delay', 'Downturned corners of mouth', 'Abnormal blood glucose concentration', 'Short nose', 'Cardiomyopathy', 'Glucose intolerance', 'Diarrhea', 'Hyperkinetic movements', 'Hyperactivity', 'Menorrhagia',  'Aplasia of the nose', 'Chronic myelogenous leukemia', 'Proteinuria', 'Microphakia', 'Laryngeal edema']
    # phenotype_list = ['Hypogonadism', 'Increased circulating ferritin concentration', 'Generalized hyperpigmentation', 'Abnormality of iron homeostasis', 'Elevated transferrin saturation', 'Hepatic fibrosis', 'Abnormal liver morphology', 'Abnormality of pancreas physiology', 'Abnormality of endocrine pancreas physiology', 'Overbite', 'Intestinal bleeding','Vitreomacular adhesion', 'Gangrene', 'Abnormality of the outer ear', 'Lissencephaly', 'Oppositional defiant disorder']
    # gene_name = "HYAL1" 
    phenotype_list = ['Glucose intolerance', 'Diarrhea']
    gene_name = "ABCC8"
    # gene_name = "HAMP"
    
    main(phenotype_list, gene_name, df_nm, df_el)