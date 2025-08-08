"""
Drosophila Connectome Interactive GUI

This module provides an interactive Jupyter widget interface for running
Infomap analysis on Drosophila connectome data with real-time parameter adjustment.
"""

import pandas as pd
import numpy as np
import os
from ipywidgets import Checkbox, VBox, Button, Output, IntSlider, HBox, HTML
from IPython.display import display

# Import our other modules - if they're not available, 
# the functions will fail with clear error messages when called
try:
    from pajek_converter import connections_to_pajek, get_connection_statistics
    from infomap_runner import run_complete_infomap_pipeline
except ImportError:
    # Functions will be imported from global namespace if available
    pass


def create_infomap_gui(csv_file, top_n_modules=100, default_output_dir="gui_output"):
    """
    Create an interactive GUI for running Infomap analysis on Drosophila connectome data.
    
    This function creates a Jupyter widget interface that allows users to:
    - Select neurotransmitter types via checkboxes
    - Set edge threshold with a slider
    - View real-time statistics about the filtered network
    - Run Infomap analysis with a single button click
    
    Parameters:
    -----------
    csv_file : str
        Path to the connections CSV file
    top_n_modules : int, default=100
        Number of top communities to show in results (currently unused but reserved for future features)
    default_output_dir : str, default="gui_output"
        Base directory name for output files
    
    Returns:
    --------
    None
        Displays interactive widget interface
    
    Raises:
    -------
    FileNotFoundError
        If the CSV file doesn't exist
    ValueError
        If the CSV file doesn't contain required columns
        
    Example:
    --------
    >>> # In a Jupyter notebook:
    >>> create_infomap_gui("connections_princeton.csv", top_n_modules=50)
    """
    # Validate input file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    try:
        # Get dataset statistics
        stats = get_connection_statistics(csv_file)
        nt_types = stats['available_neurotransmitters']
        
        # Get edge statistics for the slider
        df = pd.read_csv(csv_file)
        edge_stats = df.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum()
        min_connections = stats['min_synapses']
        max_connections = stats['max_synapses']  
        median_connections = stats['median_synapses']
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Variables to track last run (defined in this scope)
    last_combo_name = None
    last_selected_nts = None
    
    # Create checkboxes for neurotransmitter types
    checkboxes = [Checkbox(value=False, description=nt) for nt in nt_types]
    
    # Add "Select All" and "Clear All" buttons
    select_all_btn = Button(description="Select All", button_style='info', layout={'width': '120px'})
    clear_all_btn = Button(description="Clear All", button_style='warning', layout={'width': '120px'})
    
    def select_all_handler(b):
        for cb in checkboxes:
            cb.value = True
    
    def clear_all_handler(b):
        for cb in checkboxes:
            cb.value = False
    
    select_all_btn.on_click(select_all_handler)
    clear_all_btn.on_click(clear_all_handler)
    
    checkbox_controls = HBox([select_all_btn, clear_all_btn])
    checkbox_box = VBox([checkbox_controls] + checkboxes)
    
    # Create edge threshold slider  
    threshold_slider = IntSlider(
        value=1,  # Default threshold
        min=min_connections,
        max=min(max_connections, 20),  # Cap at 20 for practical UI purposes
        step=1,
        description='Min Synapses:',
        style={'description_width': 'initial'},
        continuous_update=False,
        layout={'width': '400px'}
    )
    
    # Add statistics display
    stats_html = HTML(
        value=f"""
        <div style='font-size: 12px; color: #666; margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
        <b>Dataset Statistics:</b><br>
        • Total connections: {stats['total_connections']:,}<br>
        • Unique edges: {stats['unique_edges']:,}<br>  
        • Min/Median/Max synapses: {min_connections}/{median_connections}/{max_connections}<br>
        • Available neurotransmitters: {len(nt_types)}
        </div>
        """
    )
    
    # Function to update statistics when threshold changes
    def update_stats(change):
        threshold = change['new']
        edges_above_threshold = (edge_stats >= threshold).sum()
        percentage_kept = (edges_above_threshold / len(edge_stats)) * 100
        
        stats_html.value = f"""
        <div style='font-size: 12px; color: #666; margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
        <b>Dataset Statistics:</b><br>
        • Total connections: {stats['total_connections']:,}<br>
        • Unique edges: {stats['unique_edges']:,}<br>
        • Min/Median/Max synapses: {min_connections}/{median_connections}/{max_connections}<br>
        • Available neurotransmitters: {len(nt_types)}<br>
        <br><b>With threshold ≥{threshold}:</b><br>
        • Remaining edges: {edges_above_threshold:,} ({percentage_kept:.1f}%)
        </div>
        """
    
    threshold_slider.observe(update_stats, names='value')
    
    # Advanced options
    trials_slider = IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Infomap Trials:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )
    
    # Button to generate Pajek files + run Infomap
    run_button = Button(
        description="🚀 Run Infomap Analysis", 
        button_style='success',
        layout={'width': '200px', 'height': '40px'}
    )
    output = Output()
    
    # Button callback
    def on_button_clicked(b):
        nonlocal last_combo_name, last_selected_nts
        output.clear_output()
        
        selected_nts = [cb.description for cb in checkboxes if cb.value]
        if not selected_nts:
            with output:
                print("⚠️  Please select at least one neurotransmitter.")
            return
        
        threshold = threshold_slider.value
        num_trials = trials_slider.value
        
        with output:
            print(f"🔧 Starting analysis...")
            print(f"📊 Selected: {', '.join(selected_nts)}")
            print(f"🎯 Synapse threshold: ≥{threshold}")
            print(f"🔄 Infomap trials: {num_trials}")
            print("─" * 50)
        
        try:
            # Create output directory name
            combo_name = "_".join(sorted(nt.lower().replace(" ", "") for nt in selected_nts))
            outdir = f"{default_output_dir}_{combo_name}_thresh{threshold}"
            pajek_file = f"{outdir}/{combo_name}_thresh{threshold}_graph.net"
            
            with output:
                print(f"📁 Output directory: {outdir}")
            
            # Step 1: Create pajek file with threshold
            with output:
                print(f"🔨 Converting to Pajek format...")
            
            pajek_stats = connections_to_pajek(
                csv_file=csv_file, 
                nt_types=selected_nts,
                output_file=pajek_file, 
                min_synapses=threshold
            )
            
            with output:
                print(f"✅ Network created:")
                print(f"   • Nodes: {pajek_stats['num_nodes']:,}")
                print(f"   • Edges: {pajek_stats['num_edges']:,}")
                print(f"   • Filtered out: {pajek_stats['filtered_edges']:,} edges")
                print()
            
            # Step 2: Run Infomap
            with output:
                print(f"🧠 Running Infomap analysis...")
            
            results = run_complete_infomap_pipeline(
                pajek_file=pajek_file,
                output_dir=outdir,
                num_trials=num_trials,
                parse_results=True
            )
            
            last_combo_name = combo_name
            last_selected_nts = selected_nts
            
            # Display results
            if results['infomap_result']['success']:
                with output:
                    print(f"🎉 Analysis completed successfully!")
                    print()
                    
                    if results['module_stats']:
                        module_stats = results['module_stats']
                        print(f"📈 Results Summary:")
                        print(f"   • Modules found: {module_stats['num_modules']}")
                        print(f"   • Nodes clustered: {module_stats['num_nodes']}")
                        print(f"   • Largest module: {module_stats['largest_module_size']} nodes")
                        print(f"   • Mean module size: {module_stats['mean_module_size']:.1f} nodes")
                        print()
                        
                        # Show top modules
                        print(f"🏆 Top 5 Modules by Flow:")
                        for i, module in enumerate(module_stats['modules'][:5]):
                            print(f"   {i+1}. Module {module['module_id']}: {module['num_nodes']} nodes (flow: {module['total_flow']:.3f})")
                    
                    print()
                    print(f"📂 All results saved to: {outdir}")
                    print(f"📋 Generated files: {', '.join(results['infomap_result']['output_files'])}")
                    print()
                    print("💡 Next steps:")
                    print("   • Use module_parser.parse_infomap_modules() to analyze results")  
                    print("   • Use module_visualizer functions for 3D visualization")
                    print("   • Use persistence_analysis.create_tda_gui() for topological analysis")
            else:
                with output:
                    print(f"❌ Infomap analysis failed!")
                    print(f"Error: {results['infomap_result']['stderr']}")
                    if results['infomap_result']['stdout']:
                        print(f"Output: {results['infomap_result']['stdout']}")
                        
        except Exception as e:
            with output:
                print(f"💥 Error during analysis: {str(e)}")
                print("Please check your inputs and try again.")
    
    run_button.on_click(on_button_clicked)
    
    # Create help text
    help_html = HTML(
        value="""
        <div style='font-size: 11px; color: #888; margin: 10px 0; padding: 8px; background: #f9f9f9; border-left: 3px solid #ddd;'>
        <b>💡 Tips:</b><br>
        • Higher thresholds = sparser networks, faster analysis<br>
        • More trials = better optimization, slower analysis<br>
        • Large networks may take several minutes to process<br>
        • Results include .tree files for module membership and .flow files for visualization
        </div>
        """
    )
    
    # Create the GUI layout
    main_layout = VBox([
        HTML("<h2 style='color: #333; margin-bottom: 5px;'>🧬 Drosophila Connectome Infomap Analysis</h2>"),
        HTML("<hr style='margin: 5px 0 15px 0;'>"),
        
        HTML("<h3 style='color: #555; margin: 15px 0 5px 0;'>📋 Dataset Information</h3>"),
        stats_html,
        
        HTML("<h3 style='color: #555; margin: 15px 0 5px 0;'>🧪 Neurotransmitter Selection</h3>"),
        checkbox_box,
        
        HTML("<h3 style='color: #555; margin: 15px 0 5px 0;'>⚙️ Analysis Parameters</h3>"),
        HTML("<p style='font-size: 12px; color: #666; margin: 5px 0;'>Set minimum synaptic connections for network edges (creates sparse matrix):</p>"),
        threshold_slider,
        HTML("<p style='font-size: 12px; color: #666; margin: 15px 0 5px 0;'>Number of optimization trials (more = better results, slower):</p>"),
        trials_slider,
        
        help_html,
        
        HTML("<hr style='margin: 15px 0 10px 0;'>"),
        run_button,
        HTML("<div style='margin-top: 10px;'>"),
        output,
        HTML("</div>")
    ])
    
    # Display GUI
    display(main_layout)


def quick_analysis(csv_file, nt_types, min_synapses=1, num_trials=5, output_prefix="quick"):
    """
    Run a quick Infomap analysis without the GUI (for programmatic use).
    
    Parameters:
    -----------
    csv_file : str
        Path to connections CSV file
    nt_types : list
        List of neurotransmitter types to analyze
    min_synapses : int, default=1
        Minimum synapses threshold
    num_trials : int, default=5
        Number of Infomap trials
    output_prefix : str, default="quick"
        Prefix for output directory name
        
    Returns:
    --------
    dict
        Complete analysis results
        
    Example:
    --------
    >>> results = quick_analysis(
    ...     "connections_princeton.csv",
    ...     ["dopamine", "gaba"],
    ...     min_synapses=2,
    ...     num_trials=10
    ... )
    >>> print(f"Found {results['module_stats']['num_modules']} modules")
    """
    # Create output directory
    combo_name = "_".join(sorted(nt.lower().replace(" ", "") for nt in nt_types))
    output_dir = f"{output_prefix}_{combo_name}_thresh{min_synapses}"
    pajek_file = f"{output_dir}/{combo_name}_graph.net"
    
    print(f"🔧 Running quick analysis...")
    print(f"📊 Neurotransmitters: {', '.join(nt_types)}")
    print(f"🎯 Threshold: ≥{min_synapses} synapses")
    print(f"🔄 Trials: {num_trials}")
    
    # Convert to Pajek
    pajek_stats = connections_to_pajek(
        csv_file=csv_file,
        nt_types=nt_types,
        output_file=pajek_file,
        min_synapses=min_synapses
    )
    
    print(f"📊 Network: {pajek_stats['num_nodes']} nodes, {pajek_stats['num_edges']} edges")
    
    # Run Infomap
    results = run_complete_infomap_pipeline(
        pajek_file=pajek_file,
        output_dir=output_dir,
        num_trials=num_trials
    )
    
    if results['infomap_result']['success']:
        print(f"✅ Analysis complete!")
        if results['module_stats']:
            stats = results['module_stats']
            print(f"📈 Found {stats['num_modules']} modules")
            print(f"📂 Results saved to: {output_dir}")
    else:
        print(f"❌ Analysis failed: {results['infomap_result']['stderr']}")
    
    return results


def batch_analysis(csv_file, nt_combinations, min_synapses=1, num_trials=5, output_base="batch"):
    """
    Run Infomap analysis on multiple neurotransmitter combinations.
    
    Parameters:
    -----------
    csv_file : str
        Path to connections CSV file
    nt_combinations : list of lists
        Each inner list contains neurotransmitter types to analyze together
    min_synapses : int, default=1
        Minimum synapses threshold
    num_trials : int, default=5
        Number of Infomap trials
    output_base : str, default="batch"
        Base name for output directories
        
    Returns:
    --------
    dict
        Results for each combination
        
    Example:
    --------
    >>> combinations = [
    ...     ["dopamine"],
    ...     ["gaba"],
    ...     ["dopamine", "gaba"],
    ...     ["acetylcholine", "serotonin"]
    ... ]
    >>> results = batch_analysis("connections_princeton.csv", combinations)
    """
    print(f"🔄 Starting batch analysis of {len(nt_combinations)} combinations...")
    
    all_results = {}
    
    for i, nt_types in enumerate(nt_combinations, 1):
        print(f"\n{'='*50}")
        print(f"📊 Analysis {i}/{len(nt_combinations)}: {', '.join(nt_types)}")
        print(f"{'='*50}")
        
        try:
            results = quick_analysis(
                csv_file=csv_file,
                nt_types=nt_types,
                min_synapses=min_synapses,
                num_trials=num_trials,
                output_prefix=f"{output_base}_{i:02d}"
            )
            
            combo_key = "_".join(sorted(nt_types))
            all_results[combo_key] = results
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            combo_key = "_".join(sorted(nt_types))
            all_results[combo_key] = {'error': str(e)}
    
    print(f"\n🎉 Batch analysis complete!")
    print(f"📋 Successfully analyzed: {sum(1 for r in all_results.values() if 'error' not in r)}/{len(nt_combinations)}")
    
    return all_results


if __name__ == "__main__":
    # Example usage when run as a script
    print("Connectome GUI - Example Usage")
    print("=" * 40)
    
    csv_file = "connections_princeton.csv"
    
    if os.path.exists(csv_file):
        print(f"CSV file found: {csv_file}")
        print("\nTo use the GUI in Jupyter:")
        print(f">>> from connectome_gui import create_infomap_gui")
        print(f">>> create_infomap_gui('{csv_file}')")
        
        print(f"\nTo run a quick analysis:")
        print(f">>> from connectome_gui import quick_analysis")
        print(f">>> results = quick_analysis('{csv_file}', ['gaba', 'dopamine'])")
        
        # Try to get basic stats
        try:
            from pajek_converter import get_connection_statistics
            stats = get_connection_statistics(csv_file)
            print(f"\nDataset info:")
            print(f"• {stats['total_connections']:,} total connections")
            print(f"• {len(stats['available_neurotransmitters'])} neurotransmitter types")
            print(f"• Available: {', '.join(stats['available_neurotransmitters'][:5])}...")
        except Exception as e:
            print(f"Could not load dataset info: {e}")
    else:
        print(f"Example CSV file '{csv_file}' not found.")
        print("Please provide the correct path to your connections file.")
