# Quick Reference Cheat Sheet

## 🚀 Installation (30 seconds)
```bash
pip install -r requirements.txt
```

## ⚡ Run Demo (5 minutes)
```bash
cd examples
python run_head_experiment.py --mode demo
```

## 📝 Basic Experiment Template
```python
from experiments import ExperimentConfig, ArchitectureExperiment

config = ExperimentConfig(
    experiment_name="my_experiment",
    num_heads=8,              # Number of attention heads
    num_layers=6,             # Number of transformer layers
    hidden_size=512,          # Hidden dimension (must divide by num_heads)
    batch_size=4,             # Training batch size
    num_epochs=2,             # Training epochs
    output_dir="./results"
)

experiment = ArchitectureExperiment(config)
results = experiment.run()
```

## 🔧 Key Configuration Options

### Architecture
```python
num_heads=12              # Attention heads (1-32)
num_layers=12             # Transformer layers (2-24)
hidden_size=768           # Must be divisible by num_heads
intermediate_size=3072    # FFN size (default: 4×hidden_size)
```

### Training
```python
batch_size=8              # Training batch size
num_epochs=3              # Number of epochs
learning_rate=5e-5        # Learning rate
train_size=1000          # Limit training examples (None=all)
use_fp16=True            # Mixed precision training
```

### Data
```python
dataset_name="wikitext"
dataset_config="wikitext-2-raw-v1"
max_seq_length=512
```

## 📊 Compare Experiments
```python
from experiments import compare_experiments, ExperimentResults

# Load results
results1 = ExperimentResults.load("./results/exp1/results.json")
results2 = ExperimentResults.load("./results/exp2/results.json")

# Compare
df = compare_experiments([results1, results2])
```

## 🎯 Common Experiment Patterns

### Pattern 1: Test Single Parameter
```python
results = []
for value in [4, 8, 12, 16]:
    config = ExperimentConfig(
        experiment_name=f"heads_{value}",
        num_heads=value,
        hidden_size=384,  # Divisible by all values
        # ... other params
    )
    results.append(ArchitectureExperiment(config).run())
compare_experiments(results)
```

### Pattern 2: Grid Search
```python
from itertools import product

results = []
for heads, layers in product([4, 8], [4, 6, 8]):
    config = ExperimentConfig(
        experiment_name=f"h{heads}_l{layers}",
        num_heads=heads,
        num_layers=layers,
        hidden_size=384
    )
    results.append(ArchitectureExperiment(config).run())
```

## 🐛 Troubleshooting Quick Fixes

### Out of Memory
```python
config = ExperimentConfig(
    batch_size=2,              # ⬇️ Reduce batch size
    max_seq_length=256,        # ⬇️ Reduce sequence length
    gradient_checkpointing=True,  # ✅ Enable checkpointing
    use_fp16=True              # ✅ Use mixed precision
)
```

### Hidden Size Not Divisible
```python
# ❌ Wrong
config = ExperimentConfig(num_heads=5, hidden_size=384)

# ✅ Correct
config = ExperimentConfig(num_heads=5, hidden_size=320)  # 320/5=64
```

### Quick Testing
```python
config = ExperimentConfig(
    num_epochs=1,        # Fewer epochs
    train_size=500,      # Small dataset
    val_size=100,
    num_layers=4,        # Smaller model
    hidden_size=256
)
```

## 📈 Results Location
```
results/
└── {experiment_name}/
    ├── config.json          # Configuration
    ├── results.json         # All metrics
    ├── model.pt            # Model weights
    └── training_curves.png # Plots
```

## 🔍 Key Metrics to Compare

```python
df = compare_experiments(results)

# View specific columns
df[['experiment', 'final_val_perplexity', 'total_training_time', 'params_millions']]

# Sort by performance
df.sort_values('final_val_perplexity')

# Best efficiency (low perplexity per parameter)
df['efficiency'] = df['final_val_perplexity'] / df['params_millions']
df.sort_values('efficiency')
```

## 💾 Save/Load Results
```python
# Save
results.save("./my_results.json")

# Load
from experiments import ExperimentResults
results = ExperimentResults.load("./my_results.json")

# Access metrics
print(f"Perplexity: {results.final_val_perplexity}")
print(f"Parameters: {results.total_params}")
```

## 🎨 Visualizations
```python
from experiments.analysis import plot_training_curves, generate_summary_report

# Plot individual experiment
plot_training_curves(results, save_path="curves.png")

# Generate report
report = generate_summary_report(results_list, output_path="report.txt")
```

## ⚙️ Common Hidden Size Values

For different head counts:
```
Heads=4:  256, 384, 512, 768, 1024
Heads=5:  320, 640, 960, 1280
Heads=6:  384, 768, 1152
Heads=8:  256, 512, 768, 1024
Heads=12: 384, 768, 1152
Heads=16: 512, 1024, 1536
```

## 🔗 Quick Links

- Full docs: `USAGE.md`
- Getting started: `GETTING_STARTED.md`
- Examples: `examples/`
- Tests: `python test_package.py`

## 📞 Common Commands

```bash
# Run demo
python examples/run_head_experiment.py --mode demo

# Run full comparison
python examples/run_head_experiment.py --mode full

# Run layer experiments
python examples/run_layer_experiment.py

# Test installation
python test_package.py
```

## 💡 Pro Tips

1. **Start small**: Use small models and datasets for initial testing
2. **Check divisibility**: Ensure `hidden_size % num_heads == 0`
3. **Use FP16**: Enable for 2x speedup and lower memory
4. **Save often**: Results auto-save, but check output paths
5. **Compare systematically**: Change one variable at a time

---
**Need more details?** Check `GETTING_STARTED.md` or `USAGE.md`
