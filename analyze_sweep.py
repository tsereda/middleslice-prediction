#!/usr/bin/env python3
"""
Analyze and compare results from a W&B sweep comparing model architectures.
"""

import wandb
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_sweep_runs(entity, project, sweep_id):
    """Fetch all runs from a specific sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs
    
    print(f"Found {len(runs)} runs in sweep")
    return runs


def extract_run_data(runs):
    """Extract key metrics from runs."""
    data = []
    
    for run in runs:
        if run.state != "finished":
            print(f"Skipping run {run.name} (state: {run.state})")
            continue
            
        summary = run.summary._json_dict
        config = run.config
        
        data.append({
            'run_name': run.name,
            'model_type': config.get('model_type', 'unknown'),
            'final_loss': summary.get('avg_epoch_l1_loss', np.nan),
            'min_loss': min([h.get('avg_epoch_l1_loss', np.inf) for h in run.scan_history() if 'avg_epoch_l1_loss' in h] or [np.nan]),
            'epochs': config.get('epochs', 'unknown'),
            'batch_size': config.get('batch_size', 'unknown'),
            'lr': config.get('lr', 'unknown'),
            'feature_size': config.get('feature_size', 'unknown'),
            'run_url': run.url
        })
    
    return pd.DataFrame(data)


def plot_model_comparison(df, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Final loss by model type
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_types = df['model_type'].unique()
    colors = {'swin_unetr': '#FF6B6B', 'unetr': '#4ECDC4', 'basic_unet': '#45B7D1'}
    
    for model_type in model_types:
        model_data = df[df['model_type'] == model_type]
        ax.scatter([model_type] * len(model_data), model_data['final_loss'], 
                  s=100, alpha=0.6, color=colors.get(model_type, 'gray'),
                  label=model_type)
    
    ax.set_ylabel('Final L1 Loss', fontsize=12)
    ax.set_xlabel('Model Architecture', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plot_path = output_dir / 'model_comparison.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # Plot 2: Best loss by model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_by_model = df.groupby('model_type')['min_loss'].min().sort_values()
    bars = ax.bar(range(len(best_by_model)), best_by_model.values)
    
    # Color bars
    for i, (model_type, _) in enumerate(best_by_model.items()):
        bars[i].set_color(colors.get(model_type, 'gray'))
    
    ax.set_xticks(range(len(best_by_model)))
    ax.set_xticklabels(best_by_model.index, fontsize=11)
    ax.set_ylabel('Best L1 Loss', fontsize=12)
    ax.set_title('Best Performance by Model Architecture', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(best_by_model.values):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plot_path = output_dir / 'best_by_model.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()


def print_summary(df):
    """Print text summary of results."""
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal runs analyzed: {len(df)}")
    
    print("\n--- Performance by Model ---")
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        print(f"\n{model_type.upper()}:")
        print(f"  Runs: {len(model_data)}")
        print(f"  Best loss: {model_data['min_loss'].min():.6f}")
        print(f"  Avg final loss: {model_data['final_loss'].mean():.6f}")
        print(f"  Std final loss: {model_data['final_loss'].std():.6f}")
    
    print("\n--- Overall Best Run ---")
    best_run = df.loc[df['min_loss'].idxmin()]
    print(f"  Model: {best_run['model_type']}")
    print(f"  Best loss: {best_run['min_loss']:.6f}")
    print(f"  Run name: {best_run['run_name']}")
    print(f"  URL: {best_run['run_url']}")
    
    print("\n" + "="*80)


def save_results_csv(df, output_dir):
    """Save detailed results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'sweep_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B sweep results")
    parser.add_argument('--entity', type=str, required=True, help='W&B entity (username)')
    parser.add_argument('--project', type=str, default='brats2023-architecture-comparison', 
                       help='W&B project name')
    parser.add_argument('--sweep_id', type=str, required=True, help='Sweep ID to analyze')
    parser.add_argument('--output_dir', type=str, default='./sweep_analysis', 
                       help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    print(f"Fetching sweep data from {args.entity}/{args.project}/{args.sweep_id}...")
    
    # Get runs
    runs = get_sweep_runs(args.entity, args.project, args.sweep_id)
    
    # Extract data
    df = extract_run_data(runs)
    
    if df.empty:
        print("No completed runs found!")
        return
    
    # Generate outputs
    print_summary(df)
    save_results_csv(df, args.output_dir)
    plot_model_comparison(df, args.output_dir)
    
    print(f"\n✅ Analysis complete! Check {args.output_dir}/ for outputs.")


if __name__ == '__main__':
    main()
