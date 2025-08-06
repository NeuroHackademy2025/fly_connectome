"""
Infomap Module Parser

This module parses Infomap .tree files and merges module assignments 
with neuron coordinate data for downstream analysis and visualization.
"""

import pandas as pd
import numpy as np
import os


def parse_infomap_modules(tree_file, pajek_file, coords_file, max_levels=2):
    """
    Parse Infomap tree file and merge with neuron coordinates.
    
    This function:
    1. Builds node ID to root ID mapping from Pajek file
    2. Parses .tree file to extract hierarchical module assignments  
    3. Merges module data with neuron coordinates
    4. Extracts module levels (1, 2, etc.) from hierarchical paths
    
    Parameters:
    -----------
    tree_file : str
        Path to Infomap .tree output file
    pajek_file : str
        Path to original Pajek .net file (for node ID mapping)
    coords_file : str
        Path to coordinates CSV file with columns: ['root_id', 'position']
    max_levels : int, default=2
        Maximum number of hierarchical levels to extract from module paths
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - neuron_id: Root ID of neuron
        - module_path: Full hierarchical module path (e.g., "1:2:3") 
        - level_1, level_2, ...: Module assignments at each level
        - x, y, z: 3D coordinates
        - All original columns from coords_file
        
    Raises:
    -------
    FileNotFoundError
        If any input file doesn't exist
    ValueError
        If coordinate parsing fails or required columns are missing
        
    Example:
    --------
    >>> df = parse_infomap_modules(
    ...     "output/network.tree",
    ...     "output/network.net", 
    ...     "coordinates.csv"
    ... )
    >>> print(f"Parsed {len(df)} neurons across {df['level_1'].nunique()} level-1 modules")
    """
    # Validate input files
    for file_path, name in [(tree_file, "tree"), (pajek_file, "pajek"), (coords_file, "coordinates")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{name.capitalize()} file not found: {file_path}")
    
    # Step 1: Build node ID to root ID mapping from Pajek file
    id_to_node = {}
    try:
        with open(pajek_file, "r") as f:
            in_vertices = False
            for line in f:
                line = line.strip()
                if line.startswith("*Vertices"):
                    in_vertices = True
                    continue
                if line.startswith("*Edges"):
                    break
                if in_vertices and line:
                    parts = line.split()
                    if len(parts) >= 2:
                        id_num = int(parts[0])
                        root_id = parts[1].strip('"')
                        id_to_node[id_num] = root_id
    except Exception as e:
        raise ValueError(f"Error parsing Pajek file: {e}")
    
    if not id_to_node:
        raise ValueError("No node mappings found in Pajek file")
    
    # Step 2: Parse .tree file
    tree_data = []
    try:
        with open(tree_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    module_path = parts[0]
                    node_index = int(parts[-1])  # Last column is node index
                    root_id = str(id_to_node.get(node_index, f"UNKNOWN_{node_index}"))
                    tree_data.append((root_id, module_path))
    except Exception as e:
        raise ValueError(f"Error parsing tree file: {e}")
    
    if not tree_data:
        raise ValueError("No module data found in tree file")
    
    # Create modules DataFrame
    modules_df = pd.DataFrame(tree_data, columns=["neuron_id", "module_path"])
    
    # Step 3: Extract hierarchical levels
    for level in range(1, max_levels + 1):
        level_col = f"level_{level}"
        modules_df[level_col] = modules_df["module_path"].apply(
            lambda x: _extract_module_level(x, level)
        )
    
    # Step 4: Load and merge coordinates
    try:
        positions_df = pd.read_csv(coords_file)
    except Exception as e:
        raise ValueError(f"Error reading coordinates file: {e}")
    
    # Ensure neuron_id column exists in coordinates
    if 'root_id' in positions_df.columns:
        positions_df["neuron_id"] = positions_df["root_id"].astype(str)
    elif 'neuron_id' not in positions_df.columns:
        raise ValueError("Coordinates file must have either 'root_id' or 'neuron_id' column")
    
    # Merge with coordinates
    merged = pd.merge(modules_df, positions_df, on="neuron_id", how="left")
    
    # Step 5: Parse 3D coordinates
    if 'position' in merged.columns:
        try:
            # Handle "[x y z]" format
            coord_data = merged["position"].apply(_parse_position_string)
            merged[["x", "y", "z"]] = pd.DataFrame(coord_data.tolist(), index=merged.index)
        except Exception as e:
            raise ValueError(f"Error parsing position coordinates: {e}")
    elif all(col in merged.columns for col in ['x', 'y', 'z']):
        # Coordinates already in separate columns
        pass
    else:
        raise ValueError("Coordinates file must have either 'position' column or 'x', 'y', 'z' columns")
    
    # Remove neurons without coordinates
    original_count = len(merged)
    merged = merged.dropna(subset=['x', 'y', 'z'])
    if len(merged) < original_count:
        print(f"Warning: Removed {original_count - len(merged)} neurons without coordinates")
    
    return merged


def _extract_module_level(module_path, level):
    """Extract module ID at specified hierarchical level."""
    parts = module_path.split(":")
    if level <= len(parts):
        try:
            return int(parts[level - 1])
        except ValueError:
            return np.nan
    return np.nan


def _parse_position_string(pos_str):
    """Parse position string in format '[x y z]' or 'x y z'."""
    if pd.isna(pos_str):
        return [np.nan, np.nan, np.nan]
    
    try:
        # Remove brackets and split
        clean_str = str(pos_str).strip("[]()").strip()
        coords = [float(x) for x in clean_str.split()]
        
        if len(coords) >= 3:
            return coords[:3]  # Take first 3 coordinates
        else:
            return [np.nan, np.nan, np.nan]
    except (ValueError, AttributeError):
        return [np.nan, np.nan, np.nan]


def get_module_summary(modules_df, level=1):
    """
    Get summary statistics for modules at specified level.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame returned by parse_infomap_modules()
    level : int, default=1
        Hierarchical level to analyze (1, 2, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with module statistics: ['module_id', 'size', 'percentage']
        
    Example:
    --------
    >>> summary = get_module_summary(modules_df, level=1)
    >>> print(f"Largest module has {summary['size'].max()} neurons")
    """
    level_col = f"level_{level}"
    if level_col not in modules_df.columns:
        raise ValueError(f"Level {level} not found in data. Available levels: {[col for col in modules_df.columns if col.startswith('level_')]}")
    
    # Get module sizes
    module_counts = modules_df[level_col].value_counts().reset_index()
    module_counts.columns = ['module_id', 'size']
    
    # Add percentage
    total_neurons = len(modules_df)
    module_counts['percentage'] = (module_counts['size'] / total_neurons * 100).round(2)
    
    # Sort by size descending
    module_counts = module_counts.sort_values('size', ascending=False).reset_index(drop=True)
    
    return module_counts


def filter_modules_by_size(modules_df, level=1, min_size=10):
    """
    Filter modules by minimum size threshold.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame returned by parse_infomap_modules()
    level : int, default=1
        Hierarchical level to filter
    min_size : int, default=10
        Minimum number of neurons required for a module
        
    Returns:
    --------
    tuple
        (filtered_df, large_module_ids) where:
        - filtered_df: DataFrame with only neurons in large modules
        - large_module_ids: List of module IDs that meet size threshold
        
    Example:
    --------
    >>> filtered_df, large_modules = filter_modules_by_size(modules_df, level=1, min_size=20)
    >>> print(f"Kept {len(large_modules)} modules with ≥20 neurons")
    """
    level_col = f"level_{level}"
    if level_col not in modules_df.columns:
        raise ValueError(f"Level {level} not found in data")
    
    # Get module sizes
    module_counts = modules_df[level_col].value_counts()
    large_modules = module_counts[module_counts >= min_size].index.tolist()
    
    # Filter data
    filtered_df = modules_df[modules_df[level_col].isin(large_modules)].copy()
    
    return filtered_df, large_modules


if __name__ == "__main__":
    # Example usage
    print("Module Parser - Example Usage")
    print("=" * 40)
    
    # Test with example files
    tree_file = "output/network.tree"
    pajek_file = "output/network.net"
    coords_file = "coordinates.csv"
    
    if all(os.path.exists(f) for f in [tree_file, pajek_file, coords_file]):
        try:
            # Parse modules
            modules_df = parse_infomap_modules(tree_file, pajek_file, coords_file)
            print(f"✅ Parsed {len(modules_df)} neurons")
            
            # Get summary
            summary = get_module_summary(modules_df, level=1)
            print(f"📊 Found {len(summary)} level-1 modules")
            print(f"🏆 Largest module: {summary['size'].max()} neurons")
            
            # Filter by size
            filtered_df, large_modules = filter_modules_by_size(modules_df, level=1, min_size=10)
            print(f"🔍 {len(large_modules)} modules have ≥10 neurons")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Example files not found. Please provide:")
        print("- .tree file from Infomap output")
        print("- .net Pajek file")  
        print("- coordinates.csv file")
