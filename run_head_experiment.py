"""
Example: Experimenting with Different Numbers of Attention Heads

This script demonstrates how to:
1. Run a baseline experiment with default GPT-2 architecture
2. Run experiments with modified number of attention heads
3. Compare the results
"""
import sys
sys.path.append('..')

from experiments import ExperimentConfig, ArchitectureExperiment
from experiments import compare_experiments, plot_training_curves, generate_summary_report
from pathlib import Path


def run_head_experiments():
    """Run experiments comparing different numbers of attention heads."""
    
    print("="*80)
    print("ATTENTION HEAD EXPERIMENTS")
    print("="*80)
    print("\nThis experiment will compare GPT-2 models with different numbers of")
    print("attention heads while keeping other architectural parameters constant.\n")
    
    # Define experiments to run
    head_configs = [
        {"num_heads": 12, "name": "baseline_12_heads"},
        {"num_heads": 5, "name": "modified_5_heads"},
        {"num_heads": 8, "name": "modified_8_heads"},
        {"num_heads": 16, "name": "modified_16_heads"},
    ]
    
    results_list = []
    
    # Run each experiment
    for config in head_configs:
        print(f"\n{'='*80}")
        print(f"Running experiment: {config['name']}")
        print(f"Number of heads: {config['num_heads']}")
        print(f"{'='*80}\n")
        
        # Create configuration
        exp_config = ExperimentConfig(
            experiment_name=config['name'],
            base_model="gpt2",
            num_heads=config['num_heads'],
            num_layers=6,  # Reduced for faster training
            hidden_size=384,  # Must be divisible by num_heads
            max_position_embeddings=512,
            
            # Training parameters (small scale for demo)
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            num_epochs=2,  # Quick training for demo
            warmup_steps=100,
            
            # Data parameters
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            max_seq_length=256,
            train_size=1000,  # Small subset for demo
            val_size=200,
            
            # Optimization
            use_fp16=True,
            
            # Output
            output_dir="./results/head_experiments",
            seed=42,
        )
        
        # Adjust hidden_size to be divisible by num_heads if needed
        if exp_config.hidden_size % exp_config.num_heads != 0:
            # Find closest divisible hidden size
            base_hidden = 384
            while base_hidden % exp_config.num_heads != 0:
                base_hidden += 1
            exp_config.hidden_size = base_hidden
            print(f"Adjusted hidden_size to {base_hidden} to be divisible by {exp_config.num_heads}")
        
        # Run experiment
        try:
            experiment = ArchitectureExperiment(exp_config)
            results = experiment.run()
            results_list.append(results)
            
            # Plot individual training curves
            plot_path = Path(f"./results/head_experiments/{config['name']}/training_curves.png")
            plot_training_curves(results, save_path=plot_path)
            
        except Exception as e:
            print(f"Error running experiment {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare all experiments
    if len(results_list) > 1:
        print(f"\n{'='*80}")
        print("COMPARING ALL EXPERIMENTS")
        print(f"{'='*80}\n")
        
        # Create comparison visualizations
        comparison_df = compare_experiments(
            results_list,
            save_path="./results/head_experiments/comparison.png"
        )
        
        # Generate summary report
        report = generate_summary_report(
            results_list,
            output_path="./results/head_experiments/summary_report.txt"
        )
        print("\n" + report)
        
        # Save comparison dataframe
        comparison_df.to_csv("./results/head_experiments/comparison.csv", index=False)
        print("\nComparison data saved to: ./results/head_experiments/comparison.csv")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: ./results/head_experiments/")
    print("Check the individual experiment directories for detailed results.")


def run_single_experiment_demo():
    """Run a single quick experiment as a demo."""
    
    print("="*80)
    print("SINGLE EXPERIMENT DEMO")
    print("="*80)
    print("\nRunning a single experiment with 5 attention heads...\n")
    
    # Create a simple configuration
    config = ExperimentConfig(
        experiment_name="demo_5_heads",
        base_model="gpt2",
        num_heads=5,
        num_layers=4,
        hidden_size=320,  # Divisible by 5
        
        # Small scale for quick demo
        batch_size=4,
        num_epochs=1,
        train_size=500,
        val_size=100,
        max_seq_length=128,
        
        output_dir="./results/demo",
        seed=42,
    )
    
    # Run experiment
    experiment = ArchitectureExperiment(config)
    results = experiment.run()
    
    # Show results
    plot_training_curves(results, save_path="./results/demo/training_curves.png")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"Results saved in: {config.experiment_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run attention head experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "full"],
        default="demo",
        help="Run mode: 'demo' for single quick experiment, 'full' for complete comparison"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_single_experiment_demo()
    else:
        run_head_experiments()
