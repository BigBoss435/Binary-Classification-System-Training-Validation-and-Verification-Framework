"""
Visualization script for the Comparative Study Framework
Creates diagrams to illustrate the study design
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_study_design_diagram():
    """Create a visual representation of the comparative study design."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Comparative Study Design: Control vs Enhanced V&V', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Data Source (top)
    data_box = FancyBboxPatch((3.5, 8.5), 3, 0.6, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(data_box)
    ax.text(5, 8.8, 'ISIC 2020 Dataset\n(Identical Split)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow to split
    arrow1 = FancyArrowPatch((5, 8.5), (5, 7.8), 
                            arrowstyle='->', mutation_scale=20, 
                            color='black', linewidth=2)
    ax.add_patch(arrow1)
    
    # Split box
    split_box = FancyBboxPatch((3, 7.3), 4, 0.5, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='purple', facecolor='lavender', 
                               linewidth=2, linestyle='--')
    ax.add_patch(split_box)
    ax.text(5, 7.55, 'Train (70%) | Val (15%) | Test (15%)', 
            ha='center', va='center', fontsize=9, style='italic')
    
    # Arrows to both groups
    arrow_control = FancyArrowPatch((4, 7.3), (2.5, 6.5), 
                                   arrowstyle='->', mutation_scale=20, 
                                   color='red', linewidth=2)
    arrow_enhanced = FancyArrowPatch((6, 7.3), (7.5, 6.5), 
                                    arrowstyle='->', mutation_scale=20, 
                                    color='green', linewidth=2)
    ax.add_patch(arrow_control)
    ax.add_patch(arrow_enhanced)
    
    # CONTROL GROUP (Left)
    control_box = FancyBboxPatch((0.5, 3), 4, 3.3, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='red', facecolor='mistyrose', linewidth=3)
    ax.add_patch(control_box)
    
    ax.text(2.5, 6.2, 'CONTROL GROUP', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='darkred')
    ax.text(2.5, 5.9, 'Minimal V&V', 
            ha='center', va='top', fontsize=10, style='italic')
    
    control_components = [
        '✓ Basic data loading',
        '✓ Standard augmentation',
        '✓ BCEWithLogitsLoss',
        '✓ Random sampling',
        '✓ Single-pass evaluation',
        '✗ NO data validation',
        '✗ NO EDA',
        '✗ NO fairness assessment',
        '✗ NO adversarial testing'
    ]
    
    y_pos = 5.5
    for comp in control_components:
        color = 'darkgreen' if '✓' in comp else 'darkred'
        ax.text(0.7, y_pos, comp, ha='left', va='top', fontsize=8, color=color)
        y_pos -= 0.27
    
    # ENHANCED GROUP (Right)
    enhanced_box = FancyBboxPatch((5.5, 3), 4, 3.3, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor='green', facecolor='honeydew', linewidth=3)
    ax.add_patch(enhanced_box)
    
    ax.text(7.5, 6.2, 'ENHANCED GROUP', 
            ha='center', va='top', fontsize=12, fontweight='bold', color='darkgreen')
    ax.text(7.5, 5.9, 'Comprehensive V&V', 
            ha='center', va='top', fontsize=10, style='italic')
    
    enhanced_components = [
        '✓ Great Expectations validation',
        '✓ Comprehensive EDA',
        '✓ Advanced augmentation',
        '✓ Focal Loss',
        '✓ Weighted sampling',
        '✓ Test-Time Augmentation',
        '✓ Fairness assessment',
        '✓ Adversarial robustness',
        '✓ Threshold optimization'
    ]
    
    y_pos = 5.5
    for comp in enhanced_components:
        ax.text(5.7, y_pos, comp, ha='left', va='top', fontsize=8, color='darkgreen')
        y_pos -= 0.27
    
    # Arrows to evaluation
    arrow_control_eval = FancyArrowPatch((2.5, 3), (2.5, 2.3), 
                                        arrowstyle='->', mutation_scale=20, 
                                        color='red', linewidth=2)
    arrow_enhanced_eval = FancyArrowPatch((7.5, 3), (7.5, 2.3), 
                                         arrowstyle='->', mutation_scale=20, 
                                         color='green', linewidth=2)
    ax.add_patch(arrow_control_eval)
    ax.add_patch(arrow_enhanced_eval)
    
    # Evaluation boxes
    control_eval = FancyBboxPatch((1.2, 1.5), 2.6, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='red', facecolor='white', linewidth=2)
    enhanced_eval = FancyBboxPatch((6.2, 1.5), 2.6, 0.7, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='green', facecolor='white', linewidth=2)
    ax.add_patch(control_eval)
    ax.add_patch(enhanced_eval)
    
    ax.text(2.5, 1.85, 'Test Set Evaluation\n(Basic Metrics)', 
            ha='center', va='center', fontsize=8)
    ax.text(7.5, 1.85, 'Test Set Evaluation\n(Comprehensive)', 
            ha='center', va='center', fontsize=8)
    
    # Arrows to comparison
    arrow_comp1 = FancyArrowPatch((2.5, 1.5), (4.5, 0.8), 
                                 arrowstyle='->', mutation_scale=20, 
                                 color='purple', linewidth=2)
    arrow_comp2 = FancyArrowPatch((7.5, 1.5), (5.5, 0.8), 
                                 arrowstyle='->', mutation_scale=20, 
                                 color='purple', linewidth=2)
    ax.add_patch(arrow_comp1)
    ax.add_patch(arrow_comp2)
    
    # Comparison box
    comparison_box = FancyBboxPatch((3.5, 0.1), 3, 0.6, 
                                    boxstyle="round,pad=0.1", 
                                    edgecolor='purple', facecolor='plum', linewidth=3)
    ax.add_patch(comparison_box)
    ax.text(5, 0.4, 'COMPARATIVE ANALYSIS\nMetrics | Robustness | Fairness', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='mistyrose', edgecolor='red', label='Control: Minimal V&V'),
        mpatches.Patch(facecolor='honeydew', edgecolor='green', label='Enhanced: Comprehensive V&V'),
        mpatches.Patch(facecolor='plum', edgecolor='purple', label='Comparison: Statistical Analysis')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
             ncol=3, fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('comparative_study_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Study design diagram saved: comparative_study_diagram.png")
    plt.close()


def create_metrics_comparison_template():
    """Create a template for metrics comparison visualization."""
    
    # Example data (you'll replace with actual results)
    metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    control_vals = [0.914, 0.852, 0.823, 0.789, 0.872]  # Example
    enhanced_vals = [0.937, 0.881, 0.876, 0.842, 0.924]  # Example
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, control_vals, width, label='Control', 
                    color='salmon', alpha=0.8, edgecolor='darkred')
    bars2 = ax1.bar(x + width/2, enhanced_vals, width, label='Enhanced', 
                    color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    
    ax1.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Comparison: Control vs Enhanced V&V', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.7, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Right: Improvement percentages
    improvements = [(e - c) / c * 100 for c, e in zip(control_vals, enhanced_vals)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.barh(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Relative Improvement: Enhanced vs Control', 
                  fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        x_pos = val + (0.2 if val > 0 else -0.2)
        ha = 'left' if val > 0 else 'right'
        ax2.text(x_pos, i, f'{val:+.2f}%', 
                ha=ha, va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison_template.png', dpi=300, bbox_inches='tight')
    print("✓ Metrics comparison template saved: metrics_comparison_template.png")
    plt.close()


def create_vv_components_comparison():
    """Create a visual comparison of V&V components."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    components = [
        'Data Validation',
        'Exploratory Data Analysis',
        'Advanced Augmentation',
        'Class Imbalance Handling',
        'Test-Time Augmentation',
        'Fairness Assessment',
        'Adversarial Testing',
        'Threshold Optimization'
    ]
    
    control = [0, 0, 0, 0, 0, 0, 0, 0]  # Not implemented
    enhanced = [1, 1, 1, 1, 1, 1, 1, 1]  # Implemented
    
    y_pos = np.arange(len(components))
    
    # Create stacked bars
    ax.barh(y_pos, control, label='Control (Not Implemented)', 
            color='lightcoral', edgecolor='darkred', linewidth=2)
    ax.barh(y_pos, enhanced, label='Enhanced (Implemented)', 
            color='lightgreen', edgecolor='darkgreen', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components, fontsize=10)
    ax.set_xlabel('Implementation Status', fontsize=11, fontweight='bold')
    ax.set_title('V&V Components Comparison: Control vs Enhanced', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Implemented', 'Implemented'])
    
    # Add checkmarks and X marks
    for i, (c, e) in enumerate(zip(control, enhanced)):
        if e == 1:
            ax.text(1.05, i, '✓', fontsize=16, color='darkgreen', 
                   ha='left', va='center', fontweight='bold')
        if c == 0:
            ax.text(0.05, i, '✗', fontsize=16, color='darkred', 
                   ha='left', va='center', fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('vv_components_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ V&V components comparison saved: vv_components_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Creating comparative study visualizations...\n")
    
    create_study_design_diagram()
    create_metrics_comparison_template()
    create_vv_components_comparison()
    
    print("\n✓ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  1. comparative_study_diagram.png - Study design overview")
    print("  2. metrics_comparison_template.png - Performance comparison template")
    print("  3. vv_components_comparison.png - V&V components comparison")
    print("\nUse these visualizations in your thesis/paper!")
