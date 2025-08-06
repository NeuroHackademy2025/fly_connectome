"""
Simplified Module Analysis

This module provides basic module analysis with only essential visualizations:
- Module sizes
- Composition by super_class  
- Composition by flow
- Simple ranking table
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def calculate_module_quality_scores(merged_df, module_col, min_size=50):
    """Calculate simplified quality scores focusing on size and basic classification metrics."""
    quality_scores = {}
    
    module_counts = merged_df[module_col].value_counts()
    candidate_modules = module_counts[module_counts >= min_size].index
    
    if len(candidate_modules) == 0:
        print(f"⚠️ No modules found with ≥{min_size} neurons")
        return {}
    
    total_neurons = len(merged_df)
    
    for module in candidate_modules:
        module_data = merged_df[merged_df[module_col] == module]
        module_size = len(module_data)
        
        scores = {
            'size': module_size,
            'size_score': module_size / total_neurons,
        }
        
        # Only calculate purity for super_class and flow
        classification_cols = ['super_class', 'flow']
        purity_scores = []
        
        for class_col in classification_cols:
            if class_col in module_data.columns:
                non_null_data = module_data[class_col].dropna()
                
                if len(non_null_data) > 0:
                    class_counts = non_null_data.value_counts()
                    if len(class_counts) > 0:
                        purity = class_counts.iloc[0] / len(non_null_data)
                        purity_scores.append(purity)
        
        scores['avg_purity'] = float(np.mean(purity_scores)) if purity_scores else 0.5
        scores['composite_quality'] = (0.4 * scores['size_score']) + (0.6 * scores['avg_purity'])
        
        quality_scores[module] = scores
    
    return quality_scores


def select_top_modules(quality_scores, method='composite', top_n=10):
    """Select top modules based on quality criteria."""
    if not quality_scores:
        return []
    
    if method == 'composite':
        key_func = lambda x: x[1]['composite_quality']
    elif method == 'size':
        key_func = lambda x: x[1]['size']
    elif method == 'purity':
        key_func = lambda x: x[1]['avg_purity']
    else:
        raise ValueError(f"Unknown method: {method}. Use 'composite', 'size', or 'purity'")
    
    sorted_modules = sorted(quality_scores.items(), key=key_func, reverse=True)
    return [module_id for module_id, _ in sorted_modules[:top_n]]


def create_module_size_chart(quality_scores, top_modules):
    """Create simple bar chart of module sizes."""
    modules = [m for m in top_modules if m in quality_scores]
    sizes = [quality_scores[m]['size'] for m in modules]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Module {m}" for m in modules],
            y=sizes,
            marker_color='steelblue',
            text=sizes,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Module Sizes",
        xaxis_title="Module",
        yaxis_title="Number of Neurons",
        width=1000,
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_module_composition(merged_df, top_modules, module_col, class_col):
    """
    Plot composition of modules by classification (super_class or flow only).
    """
    # Filter to specified modules
    top_module_data = merged_df[merged_df[module_col].isin(top_modules)]
    
    if class_col not in top_module_data.columns or top_module_data[class_col].notna().sum() == 0:
        print(f"⚠️ Column '{class_col}' not found or has no data")
        return None
    
    # Create composition data
    composition_data = []
    for module in sorted(top_modules):
        module_data = top_module_data[top_module_data[module_col] == module]
        if len(module_data) == 0:
            continue
            
        class_counts = module_data[class_col].value_counts()
        
        for class_type, count in class_counts.items():
            composition_data.append({
                'Module': f"Module {module}",
                'Class': str(class_type),
                'Count': count,
                'Percentage': count / len(module_data) * 100
            })
    
    if not composition_data:
        return None
    
    comp_df = pd.DataFrame(composition_data)
    
    # Create stacked percentage bar chart
    fig = px.bar(comp_df, 
                 x='Module', 
                 y='Percentage', 
                 color='Class',
                 title=f"Module Composition by {class_col}",
                 labels={'Percentage': 'Percentage of Module'},
                 category_orders={'Module': [f"Module {m}" for m in sorted(top_modules)]})
    
    fig.update_layout(
        width=1200,
        height=600,
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_module_ranking_table(quality_scores, top_n=10):
    """Create simplified ranking table with essential metrics."""
    # Sort by composite quality
    sorted_modules = sorted(quality_scores.items(), 
                          key=lambda x: x[1]['composite_quality'], reverse=True)
    
    # Create table data
    table_data = []
    for rank, (module_id, scores) in enumerate(sorted_modules[:top_n], 1):
        table_data.append({
            'Rank': rank,
            'Module': module_id,
            'Size': scores['size'],
            'Purity': f"{scores['avg_purity']:.3f}",
            'Quality Score': f"{scores['composite_quality']:.3f}"
        })
    
    df = pd.DataFrame(table_data)
    
    # Create table visualization
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='center',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Modules Ranked by Quality Score",
        height=400,
        width=700
    )
    
    return fig, df


def analyze_top_modules(merged_df, level='level_1', min_size=50, top_n=10, 
                       selection_method='composite', show_plots=True):
    """
    Simplified analysis with only essential visualizations: size, super_class, flow, and ranking.
    """
    print(f"🔍 ANALYZING TOP MODULES ({level.upper()})")
    print("=" * 50)
    
    # Calculate quality scores
    quality_scores = calculate_module_quality_scores(merged_df, level, min_size)
    
    if len(quality_scores) == 0:
        print("❌ No modules found above minimum size threshold")
        return None, []
    
    # Select top modules
    top_modules = select_top_modules(quality_scores, method=selection_method, top_n=top_n)
    print(f"   Selected top {len(top_modules)} modules")
    
    # Calculate summary
    total_neurons_in_top = sum(quality_scores[m]['size'] for m in top_modules)
    total_neurons = len(merged_df)
    
    print(f"   Coverage: {total_neurons_in_top/total_neurons*100:.1f}% of network")
    
    # Generate ONLY the 4 essential visualizations
    figures = []
    
    # 1. Module sizes
    fig_size = create_module_size_chart(quality_scores, top_modules)
    figures.append(("Module Sizes", fig_size))
    
    # 2. Ranking table
    fig_table, ranking_df = create_module_ranking_table(quality_scores, top_n)
    figures.append(("Module Rankings", fig_table))
    
    # 3. Super_class composition
    fig_super = plot_module_composition(merged_df, top_modules, level, 'super_class')
    if fig_super:
        figures.append(("Composition by Super Class", fig_super))
    
    # 4. Flow composition  
    fig_flow = plot_module_composition(merged_df, top_modules, level, 'flow')
    if fig_flow:
        figures.append(("Composition by Flow", fig_flow))
    
    # Show plots
    if show_plots:
        for title, fig in figures:
            print(f"\n📈 {title}")
            fig.show()
    
    # Results
    results = {
        'top_modules': top_modules,
        'quality_scores': {m: quality_scores[m] for m in top_modules},
        'ranking_table': ranking_df,
        'total_coverage': total_neurons_in_top/total_neurons*100,
        'level': level
    }
    
    return results, figures


if __name__ == "__main__":
    print("Simplified Module Quality Analyzer")
    print("=" * 40)
    print("Generates exactly 4 visualizations:")
    print("1. Module sizes (bar chart)")
    print("2. Module rankings (table)")  
    print("3. Composition by super_class")
    print("4. Composition by flow")
    print()
    print("Usage:")
    print(">>> results, figures = analyze_top_modules(merged_df, level='level_1')")
