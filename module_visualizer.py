"""
Infomap Module 3D Visualizer

This module creates interactive 3D visualizations of Infomap modules
using Plotly, with support for hierarchical levels and filtering.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ipywidgets import RadioButtons, VBox, Output, Dropdown, IntSlider, Checkbox, HBox, HTML
from IPython.display import display


def create_3d_module_plot(modules_df, level=1, min_module_size=10, title_suffix="", 
                         color_palette=None, marker_size=3, opacity=0.7):
    """
    Create interactive 3D scatter plot of neurons colored by module assignment.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame with module assignments and coordinates (from module_parser.parse_infomap_modules)
    level : int, default=1
        Hierarchical level to visualize (1, 2, etc.)
    min_module_size : int, default=10
        Minimum neurons required for a module to be displayed
    title_suffix : str, default=""
        Additional text to append to plot title
    color_palette : list, optional
        List of color codes. If None, uses default palette
    marker_size : int, default=3
        Size of scatter plot markers
    opacity : float, default=0.7
        Opacity of markers (0-1)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot
        
    Example:
    --------
    >>> fig = create_3d_module_plot(modules_df, level=1, min_module_size=20)
    >>> fig.show()
    """
    level_col = f"level_{level}"
    if level_col not in modules_df.columns:
        available = [col for col in modules_df.columns if col.startswith('level_')]
        raise ValueError(f"Level {level} not found. Available levels: {available}")
    
    # Default color palette
    if color_palette is None:
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
    
    # Filter modules by size
    level_counts = modules_df[level_col].value_counts()
    large_modules = level_counts[level_counts >= min_module_size].index
    large_modules_sorted = sorted(large_modules)
    
    if len(large_modules_sorted) == 0:
        raise ValueError(f"No modules found with ≥{min_module_size} neurons at level {level}")
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each module
    for i, module_id in enumerate(large_modules_sorted):
        cluster = modules_df[modules_df[level_col] == module_id]
        color = color_palette[i % len(color_palette)]
        
        fig.add_trace(go.Scatter3d(
            x=cluster["x"],
            y=cluster["y"], 
            z=cluster["z"],
            mode='markers',
            marker=dict(size=marker_size, opacity=opacity, color=color),
            name=f"Module {module_id}",
            hovertemplate=(
                "<b>Module " + str(module_id) + "</b><br>"
                "Neurons: " + str(len(cluster)) + "<br>"
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "Z: %{z}<br>"
                "<extra></extra>"
            )
        ))
    
    # Update layout
    total_modules = len(modules_df[level_col].dropna().unique())
    filtered_modules = len(large_modules_sorted)
    
    title = f"Level {level} Modules ({filtered_modules}/{total_modules} modules ≥{min_module_size} neurons)"
    if title_suffix:
        title += f"<br>{title_suffix}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate', 
            zaxis_title='Z Coordinate',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
            aspectmode='data',  # Preserve natural proportions
           # #aspectratio=dict(x=1, y=1, z=1)  # Force equal aspect ratio
        ),
        legend=dict(
            title="Modules",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.01
        ),
        width=1200,  # Increased width for better visibility
        height=900,  # Increased height
        margin=dict(l=0, r=150, b=0, t=80)  # Added right margin for legend
    )
    
    return fig


def create_interactive_module_viewer(modules_df, neurotransmitters=None, threshold=None, 
                                   max_levels=None, default_min_size=10):
    """
    Create interactive widget interface for exploring module visualizations.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame with module assignments and coordinates
    neurotransmitters : list, optional
        List of neurotransmitter names for display
    threshold : int, optional
        Synapse threshold value for display
    max_levels : int, optional
        Maximum number of levels available (auto-detected if None)
    default_min_size : int, default=10
        Default minimum module size
        
    Returns:
    --------
    None
        Displays interactive widget interface
        
    Example:
    --------
    >>> create_interactive_module_viewer(
    ...     modules_df, 
    ...     neurotransmitters=['gaba', 'dopamine'],
    ...     threshold=5
    ... )
    """
    # Auto-detect available levels
    if max_levels is None:
        level_cols = [col for col in modules_df.columns if col.startswith('level_')]
        max_levels = len(level_cols)
    
    if max_levels == 0:
        raise ValueError("No hierarchical levels found in data")
    
    # Create controls
    level_options = [f"Level {i}" for i in range(1, max_levels + 1)]
    level_toggle = RadioButtons(
        options=level_options,
        value="Level 1",
        description='Hierarchy:',
        style={'description_width': 'initial'}
    )
    
    # Size filter slider
    all_sizes = []
    for i in range(1, max_levels + 1):
        level_col = f"level_{i}"
        if level_col in modules_df.columns:
            sizes = modules_df[level_col].value_counts()
            all_sizes.extend(sizes.values)
    
    max_size = max(all_sizes) if all_sizes else 100
    
    size_slider = IntSlider(
        value=default_min_size,
        min=1,
        max=min(max_size, 100),
        step=1,
        description='Min Size:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    # Display options
    marker_size_slider = IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description='Marker Size:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    opacity_slider = IntSlider(
        value=70,
        min=10,
        max=100,
        step=10,
        description='Opacity %:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    # Output widget
    plot_output = Output()
    
    # Info display
    info_html = HTML()
    
    def update_plot():
        """Update the plot based on current widget values"""
        level_str = level_toggle.value  # "Level 1", "Level 2", etc.
        level = int(level_str.split()[-1])
        min_size = size_slider.value
        marker_size = marker_size_slider.value
        opacity = opacity_slider.value / 100.0
        
        # Create title suffix
        title_parts = []
        if neurotransmitters:
            title_parts.append(f"NTs: {', '.join(neurotransmitters)}")
        if threshold is not None:
            title_parts.append(f"Threshold: ≥{threshold}")
        title_suffix = " | ".join(title_parts)
        
        with plot_output:
            plot_output.clear_output(wait=True)
            try:
                fig = create_3d_module_plot(
                    modules_df,
                    level=level,
                    min_module_size=min_size,
                    title_suffix=title_suffix,
                    marker_size=marker_size,
                    opacity=opacity
                )
                fig.show()
                
                # Update info
                level_col = f"level_{level}"
                level_counts = modules_df[level_col].value_counts()
                large_modules = level_counts[level_counts >= min_size]
                
                info_text = f"""
                <div style='font-size: 12px; color: #666; margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
                <b>Module Statistics (Level {level}):</b><br>
                • Total modules: {len(level_counts)}<br>
                • Displayed modules: {len(large_modules)} (≥{min_size} neurons)<br>
                • Total neurons: {len(modules_df):,}<br>
                • Displayed neurons: {large_modules.sum():,}<br>
                • Largest module: {level_counts.max()} neurons
                </div>
                """
                info_html.value = info_text
                
            except Exception as e:
                print(f"❌ Error creating plot: {e}")
    
    # Set up callbacks
    def on_change(change):
        update_plot()
    
    level_toggle.observe(on_change, names='value')
    size_slider.observe(on_change, names='value')
    marker_size_slider.observe(on_change, names='value')
    opacity_slider.observe(on_change, names='value')
    
    # Create layout
    controls_row1 = HBox([level_toggle, size_slider])
    controls_row2 = HBox([marker_size_slider, opacity_slider])
    
    layout = VBox([
        HTML("<h3>📊 Interactive Module Viewer</h3>"),
        controls_row1,
        controls_row2,
        info_html,
        plot_output
    ])
    
    # Display initial plot
    update_plot()
    
    # Display interface
    display(layout)


def plot_module_size_distribution(modules_df, level=1, bins=20):
    """
    Create histogram of module size distribution.
    
    Parameters:
    -----------
    modules_df : pandas.DataFrame
        DataFrame with module assignments
    level : int, default=1
        Hierarchical level to analyze
    bins : int, default=20
        Number of histogram bins
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Histogram plot
        
    Example:
    --------
    >>> fig = plot_module_size_distribution(modules_df, level=1)
    >>> fig.show()
    """
    level_col = f"level_{level}"
    if level_col not in modules_df.columns:
        raise ValueError(f"Level {level} not found in data")
    
    # Get module sizes
    module_sizes = modules_df[level_col].value_counts().values
    
    # Create histogram
    fig = go.Figure(data=[
        go.Histogram(
            x=module_sizes,
            nbinsx=bins,
            marker_color='skyblue',
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title=f"Module Size Distribution (Level {level})",
        xaxis_title="Number of Neurons per Module",
        yaxis_title="Number of Modules",
        width=800,
        height=500
    )
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Module Visualizer - Example Usage")
    print("=" * 40)
    
    # This would typically be used with data from module_parser
    print("To use this module:")
    print("1. First parse your Infomap results:")
    print("   from module_parser import parse_infomap_modules")
    print("   modules_df = parse_infomap_modules(tree_file, pajek_file, coords_file)")
    print()
    print("2. Create visualizations:")
    print("   from module_visualizer import create_3d_module_plot, create_interactive_module_viewer")
    print("   fig = create_3d_module_plot(modules_df, level=1)")
    print("   create_interactive_module_viewer(modules_df)")
