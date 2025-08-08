"""
Topological Data Analysis (TDA) for Drosophila Connectome Modules

This module performs persistence homology analysis on Infomap community detection results.
It integrates with the existing GUI workflow and uses outputs from previous Infomap analysis.
Extracts H1 homology groups and computes persistence diagrams with neuron tracking.

Key Functions:
--------------
• create_tda_gui(): Interactive GUI for TDA analysis
• quick_tda_analysis(): Programmatic TDA analysis
• plot_persistence_diagram(): Visualization of results

Usage Examples:
---------------
# After running Infomap analysis and parsing modules:
>>> from persistence_analysis import create_tda_gui, quick_tda_analysis
>>> 
>>> # Interactive GUI approach:
>>> create_tda_gui(modules_df, output_dir, nt_types, threshold)
>>> 
>>> # Programmatic approach:
>>> result, fig = quick_tda_analysis(modules_df, output_dir, ['gaba'], 5, 'count', 100)
>>> fig.show()

Requirements:
-------------
• gudhi: pip install gudhi
• Standard data science stack (numpy, pandas, plotly, ipywidgets)
"""

import numpy as np 
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import VBox, HBox, IntSlider, Dropdown, Button, Output, HTML, RadioButtons
from IPython.display import display

try:
    import gudhi as gd
except ImportError:
    gd = None

def select_well_connected_neurons(connectome_data, modules_df, count):
    """
    Select the most well-connected neurons from the modules dataset.
    
    Parameters:
    -----------
    connectome_data : pandas.DataFrame
        Connectome data with pre_root_id, post_root_id columns
    modules_df : pandas.DataFrame
        Modules dataframe with root_id column
    count : int
        Number of neurons to select
        
    Returns:
    --------
    list
        List of selected neuron IDs (most connected first)
    """
    # Count connections per neuron
    neuron_connections = {}
    for _, row in connectome_data.iterrows():
        pre = row['pre_root_id']
        post = row['post_root_id']
        neuron_connections[pre] = neuron_connections.get(pre, 0) + 1
        neuron_connections[post] = neuron_connections.get(post, 0) + 1

    # Get all neurons in modules
    all_module_neurons = set(modules_df['root_id'].unique())
    
    # Select most connected neurons that are also in modules
    connected_module_neurons = []
    for neuron_id, conn_count in sorted(neuron_connections.items(), key=lambda x: x[1], reverse=True):
        if neuron_id in all_module_neurons:
            connected_module_neurons.append(neuron_id)
        if len(connected_module_neurons) >= count:
            break
    
    # If we don't have enough well-connected neurons, fill with remaining neurons
    if len(connected_module_neurons) < count:
        remaining_neurons = list(all_module_neurons - set(connected_module_neurons))
        connected_module_neurons.extend(remaining_neurons[:count - len(connected_module_neurons)])
    
    return connected_module_neurons[:count]


def get_module_neurons(modules_df, module_id, level=1):
    """
    Extract neurons belonging to a specific module from Infomap results.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame from module_parser.parse_infomap_modules()
    module_id : int
        ID of the module to extract
    level : int, default=1
        Hierarchical level to use for module selection
        
    Returns:
    --------
    list
        List of neuron IDs (root_id) in the specified module
        
    Example:
    --------
    >>> neurons = get_module_neurons(modules_df, module_id=1, level=1)
    >>> print(f"Module 1 contains {len(neurons)} neurons")
    """
    level_col = f"level_{level}"
    if level_col not in modules_df.columns:
        raise ValueError(f"Level {level} not found in modules data. Available levels: "
                        f"{[col for col in modules_df.columns if col.startswith('level_')]}")
    
    module_neurons = modules_df[modules_df[level_col] == module_id]['root_id'].tolist()
    return module_neurons


def load_connectome_data_from_infomap_output(output_dir, nt_types, threshold):
    """
    Load the original connectome data that was used for Infomap analysis.
    
    Parameters:
    -----------
    output_dir : str
        Output directory from Infomap analysis
    nt_types : list
        List of neurotransmitter types that were analyzed
    threshold : int
        Synapse threshold that was used
        
    Returns:
    --------
    pandas.DataFrame
        Original connectome data filtered by the same criteria used in Infomap
        
    Example:
    --------
    >>> connectome = load_connectome_data_from_infomap_output("output_gaba_thresh5", ["gaba"], 5)
    """
    # Try to find the original CSV file - look in common locations
    possible_files = [
        "connections_princeton.csv",
        "../connections_princeton.csv", 
        "../../connections_princeton.csv"
    ]
    
    connectome_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            connectome_file = file_path
            break
    
    if connectome_file is None:
        raise FileNotFoundError(
            "Could not find connections_princeton.csv. Please ensure it's in the current directory."
        )
    
    # Load and filter the same way as the original analysis
    df = pd.read_csv(connectome_file)
    
    # Filter by neurotransmitter types (handle case sensitivity)
    # Convert both to uppercase for comparison
    nt_types_upper = [nt.upper() for nt in nt_types]
    filtered_df = df[df['nt_type'].isin(nt_types_upper)]
    
    # Group by pre/post pairs and sum synaptic connections (same as pajek_converter)
    edges_df = (
        filtered_df.groupby(['pre_root_id', 'post_root_id'])['syn_count']
        .sum()
        .reset_index()
    )
    
    # Apply threshold
    edges_df = edges_df[edges_df['syn_count'] >= threshold]
    
    # Convert back to connection format for TDA analysis
    connectome_data = []
    for _, row in edges_df.iterrows():
        connectome_data.append({
            'pre_root_id': row['pre_root_id'],
            'post_root_id': row['post_root_id'], 
            'syn_count': row['syn_count'],
            'nt_type': 'combined'  # Since we already filtered and combined
        })
    
    return pd.DataFrame(connectome_data)


def construct_connectivity_matrix(connectome_data, neuron_subset=None):
    """
    Constructs a connectivity matrix from connectome data for TDA analysis.
    Simplified version that works with filtered Infomap data.

    Parameters:
    -----------
    connectome_data : pandas.DataFrame
        DataFrame containing neuron connections with cols 'pre_root_id', 'post_root_id', 'syn_count'
    neuron_subset : list, optional
        List of neuron IDs to include. If None, uses all neurons in connectome_data.

    Returns:
    --------
    tuple
        (connectivity_matrix, neuron_mapping) where:
        - connectivity_matrix: numpy array representing connections
        - neuron_mapping: dict mapping matrix indices to neuron IDs
    """
    # Get all unique neurons from the connectome data
    if neuron_subset is not None:
        all_neurons = np.array(neuron_subset)
        print(f"  - Using subset of {len(all_neurons)} neurons")
        # Filter connectome data to only include subset neurons
        connectome_data = connectome_data[
            (connectome_data['pre_root_id'].isin(neuron_subset)) & 
            (connectome_data['post_root_id'].isin(neuron_subset))
        ].copy()
        print(f"  - Filtered to {len(connectome_data)} connections within subset")
    else:
        all_neurons = np.unique(np.concatenate([
            connectome_data['pre_root_id'].unique(),
            connectome_data['post_root_id'].unique()
        ]))
    
    size = len(all_neurons)
    connectivity_matrix = np.zeros((size, size), dtype=int)
    
    # Create mapping from neuron ID to matrix index
    neuron_to_index = {neuron: idx for idx, neuron in enumerate(all_neurons)}
    neuron_mapping = {idx: neuron for idx, neuron in enumerate(all_neurons)}
    
    # Fill connectivity matrix
    for _, row in connectome_data.iterrows():
        pre_neuron = row['pre_root_id']
        post_neuron = row['post_root_id']
        syn_count = int(row['syn_count'])
        
        # Skip if neurons not in our subset
        if pre_neuron not in neuron_to_index or post_neuron not in neuron_to_index:
            continue
            
        # Get matrix indices
        i = neuron_to_index[pre_neuron]
        j = neuron_to_index[post_neuron]
        
        # Use positive weights (already filtered by neurotransmitter in previous steps)
        connectivity_matrix[i, j] = syn_count
    
    return connectivity_matrix, neuron_mapping


def compute_persistence(connectivity_matrix, neuron_mapping):
    """
    Computes H1 persistence diagrams from connectivity matrix with neuron tracking.
    
    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        Connectivity matrix with edge weights
    neuron_mapping : dict
        Mapping from matrix indices to neuron IDs
    
    Returns:
    --------
    dict
        Dictionary containing persistence results and neuron tracking information
    """
    if gd is None:
        raise ImportError("GUDHI library not found. Install with: pip install gudhi")
    
    print(f"  - Processing connectivity matrix of size {connectivity_matrix.shape}")
    
    # Create distance matrix from connectivity (convert to distances)
    abs_matrix = np.abs(connectivity_matrix)
    max_weight = int(np.max(abs_matrix)) if np.max(abs_matrix) > 0 else 1
    distance_matrix = (max_weight - abs_matrix).astype(int)
    # Ensure no zero distances
    distance_matrix = np.where(distance_matrix == 0, 1, distance_matrix)
    
    print(f"  - Created distance matrix (max weight: {max_weight})")
    
    # Create simplex tree
    simplex_tree = gd.SimplexTree()
    
    # Add vertices
    for i in range(len(distance_matrix)):
        simplex_tree.insert([i], filtration=0.0)
    
    # Add edges with filtration values
    edge_count = 0
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            if distance_matrix[i][j] > 0:
                simplex_tree.insert([i, j], filtration=float(distance_matrix[i][j]))
                edge_count += 1
    
    print(f"  - Added {len(distance_matrix)} vertices and {edge_count} edges")
    
    # Expand complex to include triangles
    print("  - Expanding complex to include higher-dimensional simplices...")
    simplex_tree.expansion(2)
    
    print(f"  - Complex has {simplex_tree.num_simplices()} total simplices")
    
    # Compute persistence
    print("  - Computing persistence homology...")
    persistence = simplex_tree.persistence()
    
    # Filter H1 homologies (dimension 1)
    h1_persistence = []
    homology_tracking = []
    
    for dim, (birth, death) in persistence:
        if dim == 1:  # H1 homology
            birth_val = float(birth)
            death_val = float(death) if death != float('inf') else np.inf
            persistence_val = death_val - birth_val if death_val != np.inf else np.inf
            
            h1_persistence.append((birth_val, death_val))
            
            # Basic tracking (simplified for integration)
            homology_info = {
                'birth_time': birth_val,
                'death_time': death_val,
                'persistence': persistence_val,
                'birth_neurons': [],  # Could be enhanced to track actual neurons
                'death_neurons': []
            }
            homology_tracking.append(homology_info)
    
    print(f"  - Found {len(h1_persistence)} H1 homology groups")
    
    return {
        'persistence_diagram': h1_persistence,
        'homology_tracking': homology_tracking,
        'neuron_mapping': neuron_mapping,
        'connectivity_matrix': connectivity_matrix
    }


def plot_persistence_diagram(persistence_result):
    """
    Create interactive persistence diagram using Plotly.
    
    Parameters:
    -----------
    persistence_result : dict
        Result from compute_persistence()
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive persistence diagram
    """
    persistence_diagram = persistence_result['persistence_diagram']
    
    if len(persistence_diagram) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No H1 homology groups found",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Persistence Diagram (H1)")
        return fig
    
    # Extract birth and death times
    births = [birth for birth, death in persistence_diagram]
    deaths = [death if death != np.inf else max(births) * 1.2 for birth, death in persistence_diagram]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add points for finite persistence
    finite_mask = [death != np.inf for birth, death in persistence_diagram]
    if any(finite_mask):
        finite_births = [b for b, mask in zip(births, finite_mask) if mask]
        finite_deaths = [d for d, mask in zip(deaths, finite_mask) if mask]
        
        fig.add_trace(go.Scatter(
            x=finite_births,
            y=finite_deaths,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Finite Persistence',
            hovertemplate='Birth: %{x}<br>Death: %{y}<br>Persistence: %{customdata}<extra></extra>',
            customdata=[d - b for b, d in zip(finite_births, finite_deaths)]
        ))
    
    # Add points for infinite persistence
    infinite_mask = [death == np.inf for birth, death in persistence_diagram]
    if any(infinite_mask):
        infinite_births = [b for b, mask in zip(births, infinite_mask) if mask]
        max_death = max(deaths) if deaths else 1
        
        fig.add_trace(go.Scatter(
            x=infinite_births,
            y=[max_death] * len(infinite_births),
            mode='markers',
            marker=dict(size=10, color='red', symbol='star'),
            name='Infinite Persistence',
            hovertemplate='Birth: %{x}<br>Death: ∞<br>Persistence: ∞<extra></extra>'
        ))
    
    # Add diagonal line
    max_val = max(max(births) if births else 0, max([d for d in deaths if d != np.inf]) if deaths else 0)
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='y = x',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="H1 Persistence Diagram",
        xaxis_title="Birth Time",
        yaxis_title="Death Time",
        width=600,
        height=600,
        showlegend=True
    )
    
    return fig


def create_tda_gui(modules_df, output_dir, nt_types, threshold):
    """
    Create interactive GUI for running TDA analysis on Infomap results.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame from module_parser.parse_infomap_modules()
    output_dir : str
        Output directory from Infomap analysis (for loading connectome data)
    nt_types : list
        Neurotransmitter types used in original analysis
    threshold : int
        Synapse threshold used in original analysis
        
    Returns:
    --------
    None
        Displays interactive widget interface
    """
    if gd is None:
        error_html = HTML(
            "<div style='color: red; padding: 20px; border: 2px solid red; border-radius: 5px;'>"
            "<h3>⚠️ GUDHI Library Required</h3>"
            "<p>TDA analysis requires the GUDHI library. Install it with:</p>"
            "<code>pip install gudhi</code>"
            "</div>"
        )
        display(error_html)
        return
    
    # Get available modules and levels
    available_levels = [col for col in modules_df.columns if col.startswith('level_')]
    if not available_levels:
        error_html = HTML(
            "<div style='color: red; padding: 10px;'>"
            "No module levels found in data. Please run Infomap analysis first."
            "</div>"
        )
        display(error_html)
        return
    
    max_level = len(available_levels)
    
    # Create subset selection controls
    subset_type = RadioButtons(
        options=[
            ('Number of neurons', 'count'),
            ('Specific module', 'module')
        ],
        value='count',
        description='Subset by:'
    )
    
    # Neuron count slider - allow selection from 10 to all neurons in dataset
    unique_neurons = len(modules_df['root_id'].unique())
    neuron_count_slider = IntSlider(
        value=min(100, unique_neurons),
        min=10,
        max=unique_neurons,
        step=10,
        description='# Neurons:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )
    
    # Module selection
    level_dropdown = Dropdown(
        options=[(f"Level {i}", i) for i in range(1, max_level + 1)],
        value=1,
        description='Level:'
    )
    
    # Get modules for level 1 initially
    level_1_modules = sorted(modules_df['level_1'].dropna().unique())
    module_dropdown = Dropdown(
        options=[(f"Module {m}", m) for m in level_1_modules],
        value=level_1_modules[0] if level_1_modules else None,
        description='Module:'
    )
    
    def update_module_options(change):
        level = change['new']
        level_col = f"level_{level}"
        modules = sorted(modules_df[level_col].dropna().unique())
        module_dropdown.options = [(f"Module {m}", m) for m in modules]
        if modules:
            module_dropdown.value = modules[0]
    
    level_dropdown.observe(update_module_options, names='value')
    
    # Analysis button
    run_button = Button(
        description="🔬 Run TDA Analysis",
        button_style='success',
        layout={'width': '200px', 'height': '40px'}
    )
    
    # Output area
    output = Output()
    
    def on_run_clicked(b):
        with output:
            output.clear_output(wait=True)
            
            try:
                print("🔬 STARTING TDA ANALYSIS")
                print("=" * 40)
                
                # Load connectome data
                print("Loading connectome data...")
                connectome_data = load_connectome_data_from_infomap_output(
                    output_dir, nt_types, threshold
                )
                print(f"Loaded {len(connectome_data)} connections")
                
                # Determine neuron subset
                neuron_subset = None
                if subset_type.value == 'count':
                    # Select well-connected neurons for better TDA results
                    print(f"Selecting {neuron_count_slider.value} well-connected neurons...")
                    neuron_subset = select_well_connected_neurons(
                        connectome_data, modules_df, neuron_count_slider.value
                    )
                    print(f"Selected {len(neuron_subset)} well-connected neurons")
                elif subset_type.value == 'module':
                    # Get neurons from specific module
                    neuron_subset = get_module_neurons(
                        modules_df, module_dropdown.value, level_dropdown.value
                    )
                    print(f"Using Module {module_dropdown.value} (Level {level_dropdown.value}): {len(neuron_subset)} neurons")
                
                if not neuron_subset:
                    print("❌ No neurons selected for analysis")
                    return
                
                # Run TDA analysis
                print("\nConstructing connectivity matrix...")
                connectivity_matrix, neuron_mapping = construct_connectivity_matrix(
                    connectome_data, neuron_subset
                )
                
                print("Computing persistence homology...")
                persistence_result = compute_persistence(connectivity_matrix, neuron_mapping)
                
                # Display results
                print(f"\n✅ ANALYSIS COMPLETE")
                print(f"Found {len(persistence_result['persistence_diagram'])} H1 homology groups")
                
                # Create and show persistence diagram
                fig = plot_persistence_diagram(persistence_result)
                fig.show()
                
                # Show summary statistics
                if persistence_result['persistence_diagram']:
                    births = [birth for birth, death in persistence_result['persistence_diagram']]
                    deaths = [death for birth, death in persistence_result['persistence_diagram'] if death != np.inf]
                    infinite_count = sum(1 for birth, death in persistence_result['persistence_diagram'] if death == np.inf)
                    
                    print(f"\n📊 SUMMARY:")
                    print(f"  • Total H1 groups: {len(persistence_result['persistence_diagram'])}")
                    print(f"  • Infinite persistence: {infinite_count}")
                    print(f"  • Birth range: {min(births):.2f} - {max(births):.2f}")
                    if deaths:
                        print(f"  • Death range: {min(deaths):.2f} - {max(deaths):.2f}")
                
            except Exception as e:
                print(f"❌ Error during analysis: {str(e)}")
                import traceback
                traceback.print_exc()
    
    run_button.on_click(on_run_clicked)
    
    # Create layout
    info_html = HTML(
        f"""
        <div style='font-size: 12px; color: #666; margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
        <b>TDA Analysis Setup:</b><br>
        • Dataset: {len(modules_df)} neurons from Infomap analysis<br>
        • Neurotransmitters: {', '.join(nt_types)}<br>
        • Synapse threshold: {threshold}<br>
        • Available levels: {max_level}
        </div>
        """
    )
    
    subset_controls = VBox([
        HTML("<h4>Select Neuron Subset:</h4>"),
        subset_type,
        VBox([neuron_count_slider], layout={'margin': '0 0 0 20px'}),
        VBox([
            HBox([level_dropdown, module_dropdown])
        ], layout={'margin': '0 0 0 20px'})
    ])
    
    # Show/hide controls based on selection
    def toggle_controls(change):
        if change['new'] == 'count':
            subset_controls.children[2].layout.display = 'block'
            subset_controls.children[3].layout.display = 'none'
        else:
            subset_controls.children[2].layout.display = 'none'
            subset_controls.children[3].layout.display = 'block'
    
    subset_type.observe(toggle_controls, names='value')
    toggle_controls({'new': subset_type.value})  # Set initial state
    
    main_layout = VBox([
        HTML("<h2 style='color: #333;'>🔬 Topological Data Analysis</h2>"),
        HTML("<hr>"),
        info_html,
        subset_controls,
        HTML("<br>"),
        run_button,
        output
    ])
    
    display(main_layout)
def run_tda_analysis(connectome_data, neuron_subset=None):
    """
    Run complete TDA analysis pipeline.
    
    Parameters:
    -----------
    connectome_data : pandas.DataFrame
        Connectome data with required columns
    neuron_subset : list, optional
        List of neuron IDs to analyze
        
    Returns:
    --------
    dict
        Results from persistence analysis
    """
    print("🔬 Running TDA Analysis...")
    
    # Construct connectivity matrix
    connectivity_matrix, neuron_mapping = construct_connectivity_matrix(
        connectome_data, neuron_subset
    )
    
    # Compute persistence
    persistence_result = compute_persistence(connectivity_matrix, neuron_mapping)
    
    print(f"✅ Found {len(persistence_result['persistence_diagram'])} H1 homology groups")
    
    return persistence_result


def quick_tda_analysis(modules_df, output_dir, nt_types, threshold, subset_type='count', subset_value=100, level=1):
    """
    Quick TDA analysis function for programmatic use.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame from module_parser.parse_infomap_modules()
    output_dir : str
        Output directory from Infomap analysis
    nt_types : list
        Neurotransmitter types used in original analysis
    threshold : int
        Synapse threshold used in original analysis
    subset_type : str, default='count'
        'count' for number of neurons, 'module' for specific module
    subset_value : int, default=100
        Either number of neurons or module ID
    level : int, default=1
        Hierarchical level for module selection (only used if subset_type='module')
        
    Returns:
    --------
    tuple
        (persistence_result, plot_figure) containing results and visualization
        
    Example:
    --------
    >>> # Analyze first 200 neurons
    >>> result, fig = quick_tda_analysis(modules_df, output_dir, ['gaba'], 5, 'count', 200)
    >>> 
    >>> # Analyze specific module
    >>> result, fig = quick_tda_analysis(modules_df, output_dir, ['gaba'], 5, 'module', 1, level=1)
    """
    if gd is None:
        raise ImportError("GUDHI library required for TDA analysis. Install with: pip install gudhi")
    
    print(f"🔬 QUICK TDA ANALYSIS")
    print("=" * 30)
    
    # Load connectome data
    print("Loading connectome data...")
    connectome_data = load_connectome_data_from_infomap_output(output_dir, nt_types, threshold)
    print(f"Loaded {len(connectome_data)} connections")
    
    # Determine neuron subset
    if subset_type == 'count':
        print(f"Selecting {subset_value} well-connected neurons...")
        neuron_subset = select_well_connected_neurons(connectome_data, modules_df, subset_value)
        print(f"Selected {len(neuron_subset)} well-connected neurons")
    elif subset_type == 'module':
        neuron_subset = get_module_neurons(modules_df, subset_value, level)
        print(f"Using Module {subset_value} (Level {level}): {len(neuron_subset)} neurons")
    else:
        raise ValueError("subset_type must be 'count' or 'module'")
    
    if not neuron_subset:
        raise ValueError("No neurons selected for analysis")
    
    # Run TDA analysis
    print("\nRunning TDA analysis...")
    persistence_result = run_tda_analysis(connectome_data, neuron_subset)
    
    # Create visualization
    fig = plot_persistence_diagram(persistence_result)
    
    print(f"\n✅ Analysis complete!")
    print(f"Found {len(persistence_result['persistence_diagram'])} H1 homology groups")
    
    return persistence_result, fig


if __name__ == "__main__":
    print("Topological Data Analysis for Drosophila Connectome")
    print("=" * 50)
    print("This module integrates with the Infomap analysis workflow.")
    print("Use create_tda_gui() after running Infomap analysis.")

