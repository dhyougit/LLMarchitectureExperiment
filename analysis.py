"""
Analysis and visualization tools for comparing experiments.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .config import ExperimentResults


def compare_experiments(
    results_list: List[ExperimentResults],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare multiple experiments and create visualizations.
    
    Args:
        results_list: List of ExperimentResults objects
        save_path: Optional path to save comparison plots
        
    Returns:
        DataFrame with comparison metrics
    """
    # Create comparison dataframe
    comparison_data = []
    
    for results in results_list:
        comparison_data.append({
            'experiment': results.experiment_name,
            'num_heads': results.config.num_heads,
            'num_layers': results.config.num_layers,
            'hidden_size': results.config.hidden_size,
            'total_params': results.total_params,
            'params_millions': results.total_params / 1e6,
            'final_train_loss': results.final_train_loss,
            'final_val_loss': results.final_val_loss,
            'final_train_perplexity': results.final_train_perplexity,
            'final_val_perplexity': results.final_val_perplexity,
            'total_training_time': results.total_training_time,
            'avg_epoch_time': results.avg_epoch_time,
            'peak_memory_mb': results.peak_memory_mb,
            'tokens_per_second': results.tokens_per_second,
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n" + "="*80)
    print("Experiment Comparison")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Create visualizations
    if len(results_list) > 1:
        _create_comparison_plots(results_list, save_path)
    
    return df


def _create_comparison_plots(
    results_list: List[ExperimentResults],
    save_path: Optional[Path] = None
):
    """Create comparison plots for multiple experiments."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation Loss Comparison
    ax = axes[0, 0]
    for results in results_list:
        ax.plot(results.val_loss_history, label=results.experiment_name, marker='o')
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Perplexity Comparison
    ax = axes[0, 1]
    for results in results_list:
        ax.plot(results.val_perplexity_history, label=results.experiment_name, marker='o')
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title('Validation Perplexity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Metrics Comparison
    ax = axes[0, 2]
    experiments = [r.experiment_name for r in results_list]
    val_perplexities = [r.final_val_perplexity for r in results_list]
    bars = ax.bar(experiments, val_perplexities, color='steelblue', alpha=0.7)
    ax.set_ylabel('Final Validation Perplexity')
    ax.set_title('Final Validation Perplexity')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Plot 4: Parameter Count Comparison
    ax = axes[1, 0]
    experiments = [r.experiment_name for r in results_list]
    params = [r.total_params / 1e6 for r in results_list]
    bars = ax.bar(experiments, params, color='coral', alpha=0.7)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Size Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M',
                ha='center', va='bottom')
    
    # Plot 5: Training Time Comparison
    ax = axes[1, 1]
    experiments = [r.experiment_name for r in results_list]
    times = [r.total_training_time / 60 for r in results_list]  # Convert to minutes
    bars = ax.bar(experiments, times, color='lightgreen', alpha=0.7)
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Total Training Time')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m',
                ha='center', va='bottom')
    
    # Plot 6: Memory Usage Comparison
    ax = axes[1, 2]
    experiments = [r.experiment_name for r in results_list]
    memory = [r.peak_memory_mb / 1024 for r in results_list]  # Convert to GB
    bars = ax.bar(experiments, memory, color='mediumpurple', alpha=0.7)
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('Peak GPU Memory Usage')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}GB',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {save_path}")
    
    plt.show()


def plot_training_curves(results: ExperimentResults, save_path: Optional[Path] = None):
    """
    Plot training curves for a single experiment.
    
    Args:
        results: ExperimentResults object
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Curves: {results.experiment_name}', fontsize=14, fontweight='bold')
    
    # Plot loss
    ax = axes[0]
    ax.plot(results.train_loss_history, label='Train Loss', marker='o')
    ax.plot(results.val_loss_history, label='Validation Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot perplexity
    ax = axes[1]
    ax.plot(results.train_perplexity_history, label='Train Perplexity', marker='o')
    ax.plot(results.val_perplexity_history, label='Validation Perplexity', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def load_results_from_directory(results_dir: Path) -> ExperimentResults:
    """
    Load experiment results from a directory.
    
    Args:
        results_dir: Path to experiment results directory
        
    Returns:
        ExperimentResults object
    """
    results_file = Path(results_dir) / "results.json"
    return ExperimentResults.load(results_file)


def generate_summary_report(
    results_list: List[ExperimentResults],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a text summary report comparing experiments.
    
    Args:
        results_list: List of ExperimentResults
        output_path: Optional path to save the report
        
    Returns:
        Report text
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EXPERIMENT COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    for i, results in enumerate(results_list, 1):
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"Experiment {i}: {results.experiment_name}")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Architecture:")
        report_lines.append(f"  - Number of Heads: {results.config.num_heads}")
        report_lines.append(f"  - Number of Layers: {results.config.num_layers}")
        report_lines.append(f"  - Hidden Size: {results.config.hidden_size}")
        report_lines.append(f"  - Total Parameters: {results.total_params:,} ({results.total_params/1e6:.2f}M)")
        report_lines.append(f"\nPerformance:")
        report_lines.append(f"  - Final Validation Loss: {results.final_val_loss:.4f}")
        report_lines.append(f"  - Final Validation Perplexity: {results.final_val_perplexity:.2f}")
        report_lines.append(f"\nEfficiency:")
        report_lines.append(f"  - Training Time: {results.total_training_time/60:.2f} minutes")
        report_lines.append(f"  - Average Epoch Time: {results.avg_epoch_time:.2f} seconds")
        report_lines.append(f"  - Peak Memory: {results.peak_memory_mb/1024:.2f} GB")
        report_lines.append(f"  - Inference Speed: {results.tokens_per_second:.0f} tokens/second")
    
    report_lines.append(f"\n{'='*80}")
    report_lines.append("SUMMARY")
    report_lines.append(f"{'='*80}")
    
    # Find best performing experiment
    best_perplexity = min(results_list, key=lambda r: r.final_val_perplexity)
    fastest_training = min(results_list, key=lambda r: r.total_training_time)
    smallest_model = min(results_list, key=lambda r: r.total_params)
    
    report_lines.append(f"Best Perplexity: {best_perplexity.experiment_name} ({best_perplexity.final_val_perplexity:.2f})")
    report_lines.append(f"Fastest Training: {fastest_training.experiment_name} ({fastest_training.total_training_time/60:.2f} minutes)")
    report_lines.append(f"Smallest Model: {smallest_model.experiment_name} ({smallest_model.total_params/1e6:.2f}M parameters)")
    
    report_text = "\n".join(report_lines)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    
    return report_text
