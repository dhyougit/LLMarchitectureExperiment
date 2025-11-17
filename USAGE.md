# Usage Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-arch-experiments

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Running Your First Experiment

```python
from experiments import ExperimentConfig, ArchitectureExperiment

# Create configuration
config = ExperimentConfig(
    experiment_name="my_first_experiment",
    num_heads=5,  # Modify number of attention heads
    num_layers=6,
    hidden_size=320,  # Must be divisible by num_heads
    batch_size=4,
    num_epochs=2,
    output_dir="./results"
)

# Run experiment
experiment = ArchitectureExperiment(config)
results = experiment.run()

# Results are automatically saved to ./results/my_first_experiment/
```

### Comparing Multiple Experiments

```python
from experiments import compare_experiments, ExperimentResults

# Load results from multiple experiments
results1 = ExperimentResults.load("./results/experiment1/results.json")
results2 = ExperimentResults.load("./results/experiment2/results.json")

# Compare
comparison_df = compare_experiments([results1, results2])
```

## Configuration Options

### Architecture Parameters

- `num_heads`: Number of attention heads (default: 12)
- `num_layers`: Number of transformer layers (default: 12)
- `hidden_size`: Hidden dimension size (default: 768)
- `intermediate_size`: FFN intermediate size (default: 4 * hidden_size)
- `max_position_embeddings`: Maximum sequence length (default: 1024)
- `vocab_size`: Vocabulary size (default: 50257)

### Training Parameters

- `batch_size`: Training batch size (default: 8)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `learning_rate`: Learning rate (default: 5e-5)
- `num_epochs`: Number of training epochs (default: 3)
- `warmup_steps`: Warmup steps (default: 500)
- `weight_decay`: Weight decay (default: 0.01)
- `max_grad_norm`: Gradient clipping (default: 1.0)

### Data Parameters

- `dataset_name`: HuggingFace dataset name (default: "wikitext")
- `dataset_config`: Dataset configuration (default: "wikitext-2-raw-v1")
- `max_seq_length`: Maximum sequence length (default: 512)
- `train_size`: Number of training examples (default: None = all)
- `val_size`: Number of validation examples (default: None = all)

## Examples

### Example 1: Varying Attention Heads

```bash
cd examples
python run_head_experiment.py --mode demo  # Quick demo
python run_head_experiment.py --mode full  # Full comparison
```

### Example 2: Varying Layer Depth

```bash
cd examples
python run_layer_experiment.py
```

### Example 3: Custom Architecture

```python
from experiments import ExperimentConfig, ArchitectureExperiment

# Create a small, efficient model
config = ExperimentConfig(
    experiment_name="tiny_efficient_model",
    num_heads=4,
    num_layers=4,
    hidden_size=256,
    intermediate_size=1024,
    
    # Fast training
    batch_size=8,
    num_epochs=1,
    train_size=500,
    
    output_dir="./results/custom"
)

experiment = ArchitectureExperiment(config)
results = experiment.run()
```

## Advanced Usage

### Loading and Analyzing Results

```python
from experiments import ExperimentResults
from experiments.analysis import plot_training_curves, generate_summary_report

# Load results
results = ExperimentResults.load("./results/my_experiment/results.json")

# Plot training curves
plot_training_curves(results, save_path="./plots/curves.png")

# Generate report
report = generate_summary_report([results], output_path="./report.txt")
print(report)
```

### Custom Evaluation

```python
from experiments import MetricsCalculator
import torch

# Load model and evaluate
model = torch.load("./results/my_experiment/model.pt")
metrics = MetricsCalculator.calculate_metrics(
    model, 
    val_dataloader,
    device="cuda"
)

print(f"Loss: {metrics['loss']:.4f}")
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

## Metrics Tracked

Each experiment automatically tracks:

- **Training Loss**: Cross-entropy loss on training data
- **Validation Loss**: Cross-entropy loss on validation data
- **Perplexity**: exp(loss) for both train and validation
- **Training Time**: Total and per-epoch training time
- **Memory Usage**: Peak GPU memory consumption
- **Model Size**: Total and trainable parameters
- **Inference Speed**: Tokens processed per second

## Tips and Best Practices

1. **Hidden Size Divisibility**: Ensure `hidden_size` is divisible by `num_heads`
2. **Small Scale Testing**: Use `train_size` and `val_size` to test on small subsets first
3. **Gradient Accumulation**: Use to simulate larger batch sizes with limited memory
4. **Mixed Precision**: Enable `use_fp16=True` for faster training and lower memory
5. **Reproducibility**: Set `seed` for reproducible experiments

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Enable `gradient_checkpointing=True`
- Use smaller model (fewer layers/smaller hidden size)

### Slow Training

- Enable `use_fp16=True`
- Reduce `train_size` for testing
- Increase `batch_size` if memory allows
- Use fewer `eval_steps`

### Poor Model Performance

- Increase `num_epochs`
- Adjust `learning_rate`
- Use more training data
- Try different architecture configurations
