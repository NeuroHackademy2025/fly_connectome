"""
Drosophila Connectome to Pajek Converter

This module converts Drosophila connectome CSV data to Pajek network format
for use with network analysis tools like Infomap.
"""

import pandas as pd
import numpy as np
import os


def connections_to_pajek(csv_file, nt_types, output_file, min_synapses=1):
    """
    Convert Drosophila connectome CSV data to Pajek format for network analysis.
    
    Parameters:
    -----------
    csv_file : str
        Path to the connections CSV file (e.g., "connections_princeton.csv")
    nt_types : list
        List of neurotransmitter types to include in the analysis
    output_file : str
        Path for the output Pajek (.net) file
    min_synapses : int, default=1
        Minimum number of synapses required for an edge (threshold for sparse matrix)
    
    Returns:
    --------
    dict
        Dictionary with statistics: 
        {
            'num_edges': int,           # Number of edges in final network
            'num_nodes': int,           # Number of nodes in final network  
            'original_edges': int,      # Number of edges before threshold
            'filtered_edges': int,      # Number of edges removed by threshold
            'output_file': str          # Path to created Pajek file
        }
    
    Raises:
    -------
    ValueError
        If invalid neurotransmitter types are provided or no edges remain after filtering
    FileNotFoundError
        If the input CSV file doesn't exist
        
    Example:
    --------
    >>> stats = connections_to_pajek(
    ...     csv_file="connections_princeton.csv",
    ...     nt_types=['dopamine', 'gaba', 'acetylcholine'],
    ...     output_file="output/connectome_graph.net",
    ...     min_synapses=2
    ... )
    >>> print(f"Created network with {stats['num_nodes']} nodes")
    """
    # Check if input file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load connection data
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Check required columns
    required_cols = ['pre_root_id', 'post_root_id', 'syn_count', 'nt_type']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Validate neurotransmitter types
    available_nts = sorted(df['nt_type'].dropna().unique())
    invalid_nts = set(nt_types) - set(available_nts)
    if invalid_nts:
        raise ValueError(f"Invalid neurotransmitter types: {invalid_nts}. Available: {available_nts}")
    
    # Filter by neurotransmitter types
    nt_df = df[df['nt_type'].isin(nt_types)]
    
    if len(nt_df) == 0:
        raise ValueError(f"No connections found for neurotransmitter types: {nt_types}")
    
    # Group by pre/post root IDs and sum synaptic connections
    edges = (
        nt_df.groupby(['pre_root_id', 'post_root_id'])['syn_count']
        .sum()
        .reset_index()
        .rename(columns={'pre_root_id': 'source', 'post_root_id': 'target', 'syn_count': 'weight'})
    )
    
    original_edge_count = len(edges)
    
    # Apply threshold filter - this creates the sparse connectivity matrix
    edges = edges[edges['weight'] >= min_synapses]
    
    if len(edges) == 0:
        raise ValueError(f"No edges remain after applying threshold of {min_synapses} synapses. "
                        f"Try a lower threshold. Original edges: {original_edge_count}")
    
    # Get all unique nodes from the filtered edges
    all_nodes = pd.Index(edges['source'].tolist() + edges['target'].tolist()).unique()
    node_to_id = {node: idx + 1 for idx, node in enumerate(all_nodes)}
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write Pajek file
    try:
        with open(output_file, "w") as f:
            # Write vertices
            f.write(f"*Vertices {len(all_nodes)}\n")
            for node in all_nodes:
                f.write(f'{node_to_id[node]} "{node}"\n')
            
            # Write edges
            f.write("*Edges\n")
            for _, row in edges.iterrows():
                f.write(f"{node_to_id[row['source']]} {node_to_id[row['target']]} {row['weight']}\n")
    
    except Exception as e:
        raise IOError(f"Error writing Pajek file: {e}")
    
    return {
        'num_edges': len(edges),
        'num_nodes': len(all_nodes),
        'original_edges': original_edge_count,
        'filtered_edges': original_edge_count - len(edges),
        'output_file': output_file
    }


def get_available_neurotransmitters(csv_file):
    """
    Get list of available neurotransmitter types from a connectome CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the connections CSV file
        
    Returns:
    --------
    list
        Sorted list of available neurotransmitter types
        
    Example:
    --------
    >>> nts = get_available_neurotransmitters("connections_princeton.csv")
    >>> print("Available neurotransmitters:", nts)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
    df = pd.read_csv(csv_file)
    return sorted(df['nt_type'].dropna().unique())


def get_connection_statistics(csv_file):
    """
    Get basic statistics about connections in the dataset.
    
    Parameters:
    -----------
    csv_file : str
        Path to the connections CSV file
        
    Returns:
    --------
    dict
        Dictionary with connection statistics
        
    Example:
    --------
    >>> stats = get_connection_statistics("connections_princeton.csv")
    >>> print(f"Total connections: {stats['total_connections']}")
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
    df = pd.read_csv(csv_file)
    
    # Edge statistics (grouped by connection pairs)
    edge_stats = df.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum()
    
    return {
        'total_connections': len(df),
        'unique_edges': len(edge_stats),
        'min_synapses': int(edge_stats.min()),
        'max_synapses': int(edge_stats.max()),
        'median_synapses': int(edge_stats.median()),
        'mean_synapses': float(edge_stats.mean()),
        'available_neurotransmitters': get_available_neurotransmitters(csv_file)
    }


if __name__ == "__main__":
    # Example usage
    print("Pajek Converter - Example Usage")
    print("=" * 40)
    
    # This would run if you execute the script directly
    csv_file = "connections_princeton.csv"
    
    try:
        # Get statistics
        stats = get_connection_statistics(csv_file)
        print(f"Dataset has {stats['total_connections']} connections")
        print(f"Available neurotransmitters: {stats['available_neurotransmitters']}")
        
        # Convert to Pajek
        result = connections_to_pajek(
            csv_file=csv_file,
            nt_types=['gaba'],
            output_file="example_output.net",
            min_synapses=1
        )
        
        print(f"Created Pajek file with {result['num_nodes']} nodes and {result['num_edges']} edges")
        
    except FileNotFoundError:
        print(f"Example CSV file '{csv_file}' not found. Please provide the correct path.")
    except Exception as e:
        print(f"Error: {e}")
