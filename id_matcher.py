"""
ID Matching and Classification Integration

This module handles matching neuron IDs between Infomap module results 
and external classification data (e.g., cell types, functional annotations).
"""

import pandas as pd
import numpy as np
import os


def diagnose_id_matching(modules_df, classification_file, connections_file=None):
    """
    Comprehensive diagnostic to identify the best strategy for matching IDs
    between module results and classification data.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame with module assignments (from module_parser.parse_infomap_modules)
    classification_file : str
        Path to CSV file with neuron classifications/annotations
    connections_file : str, optional
        Path to original connections CSV for additional validation
        
    Returns:
    --------
    str or None
        Best matching strategy name, or None if no good matches found
        
    Example:
    --------
    >>> strategy = diagnose_id_matching(modules_df, "classification.csv")
    >>> if strategy:
    ...     matched_df = apply_id_matching(modules_df, "classification.csv", strategy)
    """
    print("🔍 COMPREHENSIVE ID MATCHING DIAGNOSTIC")
    print("=" * 50)
    
    # Load classification data
    if not os.path.exists(classification_file):
        raise FileNotFoundError(f"Classification file not found: {classification_file}")
    
    try:
        classification_df = pd.read_csv(classification_file)
        print(f"✓ Loaded classification file: {len(classification_df)} entries")
    except Exception as e:
        raise ValueError(f"Error reading classification file: {e}")
    
    # Examine dataset structures
    print(f"\n📊 DATASET STRUCTURES:")
    print(f"Modules DataFrame columns: {list(modules_df.columns)}")
    print(f"Classification DataFrame columns: {list(classification_df.columns)}")
    
    # Validate required columns
    if 'root_id' not in classification_df.columns:
        raise ValueError("Classification file must have 'root_id' column")
    
    # Analyze ID formats
    print(f"\n🔍 ID FORMAT ANALYSIS:")
    
    # Find potential ID columns in modules data
    potential_id_cols = [col for col in modules_df.columns 
                        if 'id' in col.lower() or 'root' in col.lower()]
    print(f"Potential ID columns in modules data: {potential_id_cols}")
    
    for col in potential_id_cols[:3]:  # Limit to first 3 to avoid spam
        if col in modules_df.columns:
            sample_ids = modules_df[col].dropna().head(5).tolist()
            id_type = type(modules_df[col].iloc[0]) if len(modules_df) > 0 else 'unknown'
            print(f"  {col}: {sample_ids} (type: {id_type.__name__})")
    
    # Check classification IDs
    class_samples = classification_df['root_id'].head(5).tolist()
    class_type = type(classification_df['root_id'].iloc[0])
    print(f"\nClassification root_id: {class_samples} (type: {class_type.__name__})")
    
    # Test matching strategies
    print(f"\n🔄 TESTING MATCHING STRATEGIES:")
    results = {}
    
    # Strategy 1: Direct neuron_id match
    if 'neuron_id' in modules_df.columns:
        merged_ids = set(modules_df['neuron_id'].astype(str))
        class_ids = set(classification_df['root_id'].astype(str))
        matches = merged_ids.intersection(class_ids)
        results['neuron_id_str'] = len(matches)
        print(f"Strategy 1 - neuron_id as string: {len(matches)} matches")
    
    # Strategy 2: Use root_id directly
    if 'root_id' in modules_df.columns:
        merged_ids = set(modules_df['root_id'].astype(str))
        class_ids = set(classification_df['root_id'].astype(str))
        matches = merged_ids.intersection(class_ids)
        results['root_id_str'] = len(matches)
        print(f"Strategy 2 - root_id as string: {len(matches)} matches")
    
    # Strategy 3: Integer matching
    try:
        if 'root_id' in modules_df.columns:
            merged_ids = set(modules_df['root_id'].astype(int))
            class_ids = set(classification_df['root_id'].astype(int))
            matches = merged_ids.intersection(class_ids)
            results['root_id_int'] = len(matches)
            print(f"Strategy 3 - root_id as integer: {len(matches)} matches")
        else:
            print(f"Strategy 3 - root_id as integer: No root_id column")
    except (ValueError, TypeError):
        print(f"Strategy 3 - root_id as integer: Failed (conversion error)")
    
    # Strategy 4: Check original connections if provided
    if connections_file and os.path.exists(connections_file):
        try:
            connections_df = pd.read_csv(connections_file)
            print(f"\n📁 CONNECTIONS FILE CHECK:")
            print(f"   Connections file shape: {connections_df.shape}")
            
            # Check if classification IDs exist in connections
            conn_pre_ids = set(connections_df['pre_root_id'].astype(str))
            conn_post_ids = set(connections_df['post_root_id'].astype(str))
            all_conn_ids = conn_pre_ids.union(conn_post_ids)
            class_ids = set(classification_df['root_id'].astype(str))
            
            conn_class_matches = all_conn_ids.intersection(class_ids)
            print(f"   IDs in both connections and classification: {len(conn_class_matches)}")
            
            if len(conn_class_matches) > len(max(results.values())) if results else 0:
                results['connections_rebuild'] = len(conn_class_matches)
                print(f"   ⚠️ Best matches are in original connections - network filtering lost IDs!")
                
        except Exception as e:
            print(f"   Could not check connections file: {e}")
    
    # Determine recommendation
    print(f"\n💡 RECOMMENDATION:")
    if results:
        best_strategy = max(results.items(), key=lambda x: x[1])
        
        if best_strategy[1] > 0:
            print(f"✓ Best matching strategy: {best_strategy[0]} with {best_strategy[1]} matches")
            match_rate = (best_strategy[1] / len(modules_df)) * 100
            print(f"  Match rate: {match_rate:.1f}% of neurons in modules")
            
            if match_rate < 50:
                print(f"  ⚠️ Low match rate - consider checking data sources")
            
            return best_strategy[0]
    
    print(f"❌ No good matches found. Possible issues:")
    print(f"   1. Classification file is for a different dataset")
    print(f"   2. IDs need preprocessing/cleaning")
    print(f"   3. Network filtering removed too many neurons")
    print(f"   4. ID format conversion needed")
    
    return None


def apply_id_matching(modules_df, classification_file, strategy):
    """
    Apply ID matching strategy to merge classification data with modules.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame with module assignments
    classification_file : str
        Path to classification CSV file
    strategy : str
        Matching strategy from diagnose_id_matching()
        
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with classification data added
        
    Example:
    --------
    >>> merged_df = apply_id_matching(modules_df, "classification.csv", "root_id_str")
    >>> print(f"Added classifications for {merged_df['cell_type'].notna().sum()} neurons")
    """
    print(f"\n🔧 APPLYING MATCHING STRATEGY: {strategy}")
    
    if strategy == 'connections_rebuild':
        raise ValueError(
            "Strategy 'connections_rebuild' requires rebuilding the network analysis "
            "to preserve original IDs. Use rebuild_network_with_classifications() instead."
        )
    
    # Load classification data
    classification_df = pd.read_csv(classification_file)
    
    # Clean up classification data
    classification_df = classification_df.replace('', np.nan)
    
    # Apply matching strategy
    modules_copy = modules_df.copy()
    
    if strategy == 'root_id_str':
        modules_copy['match_id'] = modules_copy['root_id'].astype(str)
        classification_df['match_id'] = classification_df['root_id'].astype(str)
        
    elif strategy == 'root_id_int':
        modules_copy['match_id'] = modules_copy['root_id'].astype(int)
        classification_df['match_id'] = classification_df['root_id'].astype(int)
        
    elif strategy == 'neuron_id_str':
        modules_copy['match_id'] = modules_copy['neuron_id'].astype(str)
        classification_df['match_id'] = classification_df['root_id'].astype(str)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Merge data
    merged_df = pd.merge(modules_copy, classification_df, on='match_id', how='left')
    
    # Remove temporary matching column
    merged_df = merged_df.drop('match_id', axis=1)
    
    # Report success
    total_neurons = len(merged_df)
    
    # Count matches using different possible classification indicators
    classification_indicators = ['cell_type', 'super_class', 'class', 'type', 'flow']
    matched_neurons = 0
    
    for indicator in classification_indicators:
        if indicator in merged_df.columns:
            matched = merged_df[indicator].notna().sum()
            if matched > matched_neurons:
                matched_neurons = matched
                primary_indicator = indicator
    
    if matched_neurons > 0:
        match_rate = (matched_neurons / total_neurons) * 100
        print(f"✓ Successfully matched {matched_neurons}/{total_neurons} neurons ({match_rate:.1f}%)")
        print(f"  Primary classification column: '{primary_indicator}'")
    else:
        print(f"⚠️ Merge completed but no classification data found")
        print(f"  Available columns: {list(merged_df.columns)}")
    
    return merged_df


def get_classification_summary(merged_df):
    """
    Get summary statistics of classification data in merged DataFrame.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        DataFrame with merged module and classification data
        
    Returns:
    --------
    dict
        Summary statistics for each classification column
        
    Example:
    --------
    >>> summary = get_classification_summary(merged_df)
    >>> print(f"Cell types: {len(summary['cell_type']['counts'])}")
    """
    summary = {}
    
    # Common classification column names
    class_columns = ['cell_type', 'super_class', 'class', 'type', 'flow', 'side', 
                    'hemisphere', 'region', 'neurotransmitter']
    
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 30)
    
    for col in class_columns:
        if col in merged_df.columns:
            non_null = merged_df[col].notna().sum()
            if non_null > 0:
                counts = merged_df[col].value_counts().to_dict()
                summary[col] = {
                    'total_classified': non_null,
                    'unique_categories': len(counts),
                    'counts': counts
                }
                
                print(f"\n{col.upper()}:")
                print(f"  Classified neurons: {non_null}")
                print(f"  Categories: {len(counts)}")
                
                # Show top categories
                top_5 = list(counts.items())[:5]
                for category, count in top_5:
                    print(f"    {category}: {count}")
                
                if len(counts) > 5:
                    print(f"    ... and {len(counts) - 5} more")
    
    if not summary:
        print("No classification columns found")
        print(f"Available columns: {list(merged_df.columns)}")
    
    return summary


def diagnose_and_match(modules_df, classification_file, connections_file=None, auto_apply=True):
    """
    One-step function to diagnose and apply ID matching.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame with module assignments
    classification_file : str
        Path to classification CSV file
    connections_file : str, optional
        Path to original connections file
    auto_apply : bool, default=True
        Whether to automatically apply the best strategy
        
    Returns:
    --------
    pandas.DataFrame or None
        Merged DataFrame if successful, None if failed
        
    Example:
    --------
    >>> merged_df = diagnose_and_match(modules_df, "classification.csv")
    >>> if merged_df is not None:
    ...     summary = get_classification_summary(merged_df)
    """
    # Run diagnostic
    strategy = diagnose_id_matching(modules_df, classification_file, connections_file)
    
    if strategy and strategy != 'connections_rebuild' and auto_apply:
        # Apply the best strategy
        merged_df = apply_id_matching(modules_df, classification_file, strategy)
        
        # Show summary
        summary = get_classification_summary(merged_df)
        
        return merged_df
    
    elif strategy == 'connections_rebuild':
        print(f"\n⚠️ Automatic matching not possible.")
        print(f"The network filtering step lost the connection between original neuron IDs")
        print(f"and the final network. You'll need to rebuild the analysis pipeline")
        print(f"to preserve ID mappings throughout the process.")
        return None
    
    else:
        print(f"\n❌ No suitable matching strategy found.")
        return None


if __name__ == "__main__":
    # Example usage
    print("ID Matcher - Example Usage")
    print("=" * 40)
    
    print("To use this module:")
    print("1. First parse your modules:")
    print("   from module_parser import parse_infomap_modules")
    print("   modules_df = parse_infomap_modules(tree_file, pajek_file, coords_file)")
    print()
    print("2. Then match with classifications:")
    print("   from id_matcher import diagnose_and_match")
    print("   merged_df = diagnose_and_match(modules_df, 'classification.csv')")
    print() 
    print("3. Or do it step by step:")
    print("   strategy = diagnose_id_matching(modules_df, 'classification.csv')")
    print("   merged_df = apply_id_matching(modules_df, 'classification.csv', strategy)")
    print("   summary = get_classification_summary(merged_df)")
