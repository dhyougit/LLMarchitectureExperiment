# Getting Started

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM (16GB+ recommended for larger models)

## Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd llm-arch-experiments
```

### Step 2: Set Up Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

### Step 3: Verify Installation

```bash
python test_package.py
```

## Your First Experiment

### Running the Demo

The quickest way to get started is with the demo example:

```bash
cd examples
python run_head_experiment.py --mode demo
```

This will:
1. Create a small GPT-2 model with 5 attention heads
2. Train it on a subset of WikiText-2
3. Generate training curves and metrics
4. Save results to `./results/demo/`

### Understanding the Output

After running, you'll find:

```
results/
└── demo_5_heads/
    ├── config.json          # Experiment configuration
    ├── results.json         # All metrics and results
    ├── model.pt            # Trained model weights
    └── training_curves.png # Visualization
```

## Running Your Own Experiments

### Example 1: Compare Different Number of Heads

```python
from experiments import ExperimentConfig, ArchitectureExperiment
from experiments import compare_experiments

# Define baseline
baseline_config = ExperimentConfig(
    experiment_name="baseline_12_heads",
    num_heads=12,
    num_layers=6,
    hidden_size=384,  # Must be divisible by num_heads
    batch_size=4,
    num_epochs=2,
    train_size=500,
    output_dir="./results/my_experiments"
)

# Run baseline
baseline_exp = ArchitectureExperiment(baseline_config)
baseline_results = baseline_exp.run()

# Test with 5 heads
modified_config = baseline_config.copy()
modified_config.experiment_name = "modified_5_heads"
modified_config.num_heads = 5
modified_config.hidden_size = 320  # Adjusted to be divisible by 5

modified_exp = ArchitectureExperiment(modified_config)
modified_results = modified_exp.run()

# Compare
comparison = compare_experiments([baseline_results, modified_results])
print(comparison)
```

### Example 2: Architecture Parameter Grid Search

```python
from experiments import ExperimentConfig, ArchitectureExperiment

# Parameters to test
head_counts = [4, 8, 12]
layer_counts = [4, 6, 8]

results = []

for heads in head_counts:
    for layers in layer_counts:
        config = ExperimentConfig(
            experiment_name=f"h{heads}_l{layers}",
            num_heads=heads,
            num_layers=layers,
            hidden_size=384,  # Divisible by 4, 8, and 12
            batch_size=4,
            num_epochs=1,
            train_size=500,
            output_dir="./results/grid_search"
        )
        
        exp = ArchitectureExperiment(config)
        result = exp.run()
        results.append(result)

# Analyze all results
from experiments import compare_experiments
compare_experiments(results, save_path="./results/grid_search/comparison.png")
```

## Key Parameters to Experiment With

### Architecture Parameters

1. **Number of Attention Heads** (`num_heads`)
   - Default: 12
   - Typical range: 1-32
   - Must divide `hidden_size` evenly
   - Impact: Attention diversity vs. efficiency

2. **Number of Layers** (`num_layers`)
   - Default: 12
   - Typical range: 2-24
   - Impact: Model capacity vs. speed

3. **Hidden Size** (`hidden_size`)
   - Default: 768
   - Typical range: 128-2048
   - Must be divisible by `num_heads`
   - Impact: Model capacity and memory

4. **Intermediate Size** (`intermediate_size`)
   - Default: 4 × `hidden_size`
   - Typical range: 2× to 8× `hidden_size`
   - Impact: FFN capacity

### Training Parameters

1. **Batch Size** (`batch_size`)
   - Affects training stability and speed
   - Adjust based on GPU memory

2. **Learning Rate** (`learning_rate`)
   - Default: 5e-5
   - Critical for convergence

3. **Number of Epochs** (`num_epochs`)
   - More epochs = better convergence (if not overfitting)

## Tips for Successful Experiments

### 1. Start Small

```python
config = ExperimentConfig(
    num_layers=4,        # Fewer layers
    hidden_size=256,     # Smaller hidden size
    batch_size=4,        # Small batch
    num_epochs=1,        # Quick training
    train_size=500,      # Small dataset
    val_size=100
)
```

### 2. Ensure Divisibility

```python
# Good: 384 is divisible by 1, 2, 3, 4, 6, 8, 12
config = ExperimentConfig(
    num_heads=12,
    hidden_size=384
)

# Bad: Will raise error
config = ExperimentConfig(
    num_heads=5,
    hidden_size=384  # Not divisible by 5!
)

# Fixed:
config = ExperimentConfig(
    num_heads=5,
    hidden_size=320  # 320 / 5 = 64
)
```

### 3. Monitor GPU Memory

```python
from experiments import MetricsCalculator

# Check memory after training
peak_memory = MetricsCalculator.get_gpu_memory_usage()
print(f"Peak memory: {peak_memory:.2f} MB")
```

### 4. Use Mixed Precision for Speed

```python
config = ExperimentConfig(
    use_fp16=True,  # Faster and uses less memory
    # ...
)
```

## Next Steps

1. **Run the full comparison**: `python examples/run_head_experiment.py --mode full`
2. **Try layer experiments**: `python examples/run_layer_experiment.py`
3. **Read the usage guide**: Check `USAGE.md` for detailed documentation
4. **Customize**: Modify configs for your research questions

## Troubleshooting

### "CUDA out of memory"

**Solutions:**
- Reduce `batch_size`
- Reduce `hidden_size` or `num_layers`
- Enable `gradient_checkpointing=True`
- Reduce `max_seq_length`

### "hidden_size must be divisible by num_heads"

**Solution:**
Adjust `hidden_size` to be divisible by `num_heads`:
```python
# For 5 heads, use: 320, 640, 960, 1280...
# For 7 heads, use: 336, 672, 1008, 1344...
```

### Slow Training

**Solutions:**
- Enable `use_fp16=True`
- Increase `batch_size` (if memory allows)
- Reduce `train_size` for faster iterations
- Use fewer evaluation steps

## Getting Help

- Check `USAGE.md` for detailed documentation
- See `examples/` for more examples
- Review configuration options in `experiments/config.py`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_arch_experiments,
  title={LLM Architecture Experimentation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-arch-experiments}
}
```
