"""
Visualize results from EEG inference CSV files.
Creates scatter plots colored by super categories from COCO dataset.
python tools/visualize_results.py data/runs/sub08_run3/results_avg.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path


def load_super_categories(path):
    """Load super category list from .pth file."""
    super_cats = torch.load(path)
    print(f"Loaded {len(super_cats)} super categories")
    print(f"Unique categories: {set(super_cats)}")
    return super_cats


def create_scatter_plot(df, super_cats, save_path=None):
    """
    Create scatter plot with 2_3_EOS on x-axis and GT_3_EOS on y-axis.
    Points are colored by their super category.
    
    Args:
        df: DataFrame with columns '2_3_EOS' and 'GT_3_EOS'
        super_cats: List of super categories (one per row in df)
        save_path: Optional path to save the figure
    """
    # Normalize values to [0, 1] range for better visualization
    df_norm = df.copy()
    df_norm['2_3_EOS'] = (df['2_3_EOS'] - df['2_3_EOS'].min()) / (df['2_3_EOS'].max() - df['2_3_EOS'].min())
    df_norm['GT_3_EOS'] = (df['GT_3_EOS'] - df['GT_3_EOS'].min()) / (df['GT_3_EOS'].max() - df['GT_3_EOS'].min())
    
    # Get unique categories and assign colors
    unique_cats = sorted(set(super_cats))
    n_cats = len(unique_cats)
    
    # Use a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab10' if n_cats <= 10 else 'tab20')
    colors = [cmap(i / max(n_cats - 1, 1)) for i in range(n_cats)]
    cat_to_color = dict(zip(unique_cats, colors))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each category separately for legend
    for cat in unique_cats:
        mask = [sc == cat for sc in super_cats]
        x = df_norm.loc[mask, '2_3_EOS']
        y = df_norm.loc[mask, 'GT_3_EOS']
        ax.scatter(x, y, c=[cat_to_color[cat]], label=cat, alpha=0.6, s=50)
    
    # Add diagonal line (y=x) for reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='y=x')
    
    # Calculate and display correlation
    corr = df['2_3_EOS'].corr(df['GT_3_EOS'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    # Labels and title
    ax.set_xlabel('Stage 2-3 Cosine Similarity (EOS) [Normalized]', fontsize=12)
    ax.set_ylabel('Ground Truth - Stage 3 Cosine Similarity (EOS) [Normalized]', fontsize=12)
    ax.set_title('Caption Semantic Similarity by Super Category', fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Super Category')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, ax


def print_statistics(df, super_cats):
    """Print statistics grouped by super category."""
    df_with_cats = df.copy()
    df_with_cats['super_category'] = super_cats
    
    print("\n" + "="*60)
    print("STATISTICS BY SUPER CATEGORY")
    print("="*60)
    
    for cat in sorted(set(super_cats)):
        cat_df = df_with_cats[df_with_cats['super_category'] == cat]
        print(f"\n{cat} (n={len(cat_df)}):")
        print(f"  2_3_EOS:  mean={cat_df['2_3_EOS'].mean():.3f}, std={cat_df['2_3_EOS'].std():.3f}")
        print(f"  GT_3_EOS: mean={cat_df['GT_3_EOS'].mean():.3f}, std={cat_df['GT_3_EOS'].std():.3f}")
        print(f"  Correlation: {cat_df['2_3_EOS'].corr(cat_df['GT_3_EOS']):.3f}")
    
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"2_3_EOS:  mean={df['2_3_EOS'].mean():.3f}, std={df['2_3_EOS'].std():.3f}")
    print(f"GT_3_EOS: mean={df['GT_3_EOS'].mean():.3f}, std={df['GT_3_EOS'].std():.3f}")
    print(f"Correlation: {df['2_3_EOS'].corr(df['GT_3_EOS']):.3f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize EEG inference results')
    parser.add_argument('csv_path', type=str, help='Path to results CSV file')
    parser.add_argument('--super_cats', type=str, 
                       default='/zhome/73/b/145313/thesis/data/images/test_super_cat_list.pth',
                       help='Path to super category list .pth file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for saving the figure (optional)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Load super categories
    super_cats = load_super_categories(args.super_cats)
    
    # Verify lengths match
    if len(df) != len(super_cats):
        print(f"WARNING: CSV has {len(df)} rows but super_cats has {len(super_cats)} entries")
        print("Using the minimum length for visualization")
        min_len = min(len(df), len(super_cats))
        df = df.iloc[:min_len]
        super_cats = super_cats[:min_len]
    
    # Print statistics
    print_statistics(df, super_cats)
    
    # Create scatter plot
    if args.output is None:
        # Generate default output path
        csv_path = Path(args.csv_path)
        args.output = csv_path.parent / f"{csv_path.stem}_scatter.png"
    
    create_scatter_plot(df, super_cats, save_path=args.output)


if __name__ == "__main__":
    main()
