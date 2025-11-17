"""
Example: Experimenting with Different Numbers of Layers

This script demonstrates how to experiment with varying the number
of transformer layers while keeping other parameters constant.
"""
import sys
sys.path.append('..')

from experiments import ExperimentConfig, ArchitectureExperiment
from experiments import compare_experiments, generate_summary_report
from pathlib import Path


def run_layer_experiments():
    """Run experiments comparing different numbers of layers."""
    
    print("="*80)
    print("LAYER DEPTH EXPERIMENTS")
    print("="*80)
    print("\nComparing GPT-2 models with different numbers of layers.\n")
    
    # Define experiments
    layer_configs = [
        {"num_layers": 4, "name": "shallow_4_layers"},
        {"num_layers": 6, "name": "medium_6_layers"},
        {"num_layers": 8, "name": "deep_8_layers"},
        {"num_layers": 12, "name": "very_deep_12_layers"},
    ]
    
    results_list = []
    
    for config in layer_configs:
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"{'='*80}\n")
        
        exp_config = ExperimentConfig(
            experiment_name=config['name'],
            base_model="gpt2",
            num_heads=8,
            num_layers=config['num_layers'],
            hidden_size=512,
            
            # Training
            batch_size=4,
            gradient_accumulation_steps=2,
            num_epochs=2,
            train_size=1000,
            val_size=200,
            
            output_dir="./results/layer_experiments",
            seed=42,
        )
        
        try:
            experiment = ArchitectureExperiment(exp_config)
            results = experiment.run()
            results_list.append(results)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Compare results
    if len(results_list) > 1:
        compare_experiments(
            results_list,
            save_path="./results/layer_experiments/comparison.png"
        )
        
        generate_summary_report(
            results_list,
            output_path="./results/layer_experiments/summary_report.txt"
        )


if __name__ == "__main__":
    run_layer_experiments()
