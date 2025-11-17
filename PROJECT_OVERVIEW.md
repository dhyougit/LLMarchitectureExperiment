# LLM Architecture Experimentation Framework - Project Overview

## 📦 Package Contents

This is a complete, production-ready Python package for experimenting with transformer architecture modifications. Perfect for LLM researchers who want to understand how architectural changes impact model performance.

## 🎯 Key Features

### 1. **Easy Architecture Modification**
- Modify attention heads, layers, hidden dimensions, and more
- Built-in validation to prevent configuration errors
- Support for GPT-2 based architectures

### 2. **Comprehensive Metrics**
- Training/validation loss and perplexity
- Training time and memory usage
- Model parameter counts
- Inference speed measurements

### 3. **Experiment Management**
- Automatic result saving and versioning
- Easy experiment comparison
- Visualization tools included

### 4. **Production Ready**
- Clean, modular code structure
- Type hints and documentation
- Error handling and validation
- Ready for GitHub publication

## 📁 Package Structure

```
llm-arch-experiments/
├── experiments/           # Core package
│   ├── config.py         # Configuration management
│   ├── model.py          # Model architecture
│   ├── trainer.py        # Training orchestration
│   ├── metrics.py        # Metrics calculation
│   └── analysis.py       # Visualization & comparison
├── data/                 # Data loading utilities
│   └── dataset.py        
├── examples/             # Ready-to-run examples
│   ├── run_head_experiment.py    # Compare attention heads
│   └── run_layer_experiment.py   # Compare layer depths
├── requirements.txt      # Dependencies
├── setup.py             # Package installation
├── test_package.py      # Verification tests
├── README.md            # Project overview
├── GETTING_STARTED.md   # Quick start guide
├── USAGE.md             # Detailed documentation
└── LICENSE              # MIT License
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo (5 minutes on GPU)
cd examples
python run_head_experiment.py --mode demo

# 3. View results
ls ../results/demo_5_heads/
```

## 💡 Example Use Cases

### Use Case 1: Attention Head Analysis
**Research Question:** How does the number of attention heads affect model performance?

```python
from experiments import ExperimentConfig, ArchitectureExperiment, compare_experiments

# Test 5, 8, 12, and 16 heads
results = []
for num_heads in [5, 8, 12, 16]:
    config = ExperimentConfig(
        experiment_name=f"{num_heads}_heads",
        num_heads=num_heads,
        hidden_size=320,  # Divisible by all head counts
        num_layers=6,
        batch_size=4,
        num_epochs=2
    )
    exp = ArchitectureExperiment(config)
    results.append(exp.run())

# Compare all experiments
compare_experiments(results)
```

### Use Case 2: Model Scaling Study
**Research Question:** How does model depth affect performance vs. efficiency?

```python
# Compare shallow vs deep models
configs = {
    'shallow': {'num_layers': 4, 'hidden_size': 512},
    'medium': {'num_layers': 8, 'hidden_size': 512},
    'deep': {'num_layers': 12, 'hidden_size': 512}
}

results = []
for name, params in configs.items():
    config = ExperimentConfig(
        experiment_name=name,
        **params,
        num_epochs=3
    )
    results.append(ArchitectureExperiment(config).run())

# Analyze trade-offs
compare_experiments(results)
```

### Use Case 3: Efficient Architecture Search
**Research Question:** Find the smallest model that maintains good perplexity

```python
import itertools

heads = [4, 8]
layers = [4, 6, 8]
hidden_sizes = [256, 384, 512]

results = []
for h, l, d in itertools.product(heads, layers, hidden_sizes):
    if d % h == 0:  # Ensure divisibility
        config = ExperimentConfig(
            experiment_name=f"h{h}_l{l}_d{d}",
            num_heads=h,
            num_layers=l,
            hidden_size=d,
            num_epochs=2
        )
        results.append(ArchitectureExperiment(config).run())

# Find best trade-off
df = compare_experiments(results)
df['efficiency_score'] = df['final_val_perplexity'] / df['params_millions']
print(df.sort_values('efficiency_score'))
```

## 📊 What Gets Measured

Every experiment automatically tracks:

| Metric | Description |
|--------|-------------|
| **Loss** | Training and validation cross-entropy loss |
| **Perplexity** | Exponential of loss (interpretability) |
| **Training Time** | Total and per-epoch timing |
| **Memory Usage** | Peak GPU memory consumption |
| **Model Size** | Total and trainable parameters |
| **Inference Speed** | Tokens processed per second |

## 🎨 Visualization Examples

The framework automatically generates:

1. **Training Curves**: Loss and perplexity over time
2. **Comparison Plots**: Side-by-side bar charts
3. **Summary Tables**: Comprehensive metric comparison
4. **Text Reports**: Detailed analysis documents

## 🔬 Supported Modifications

### Architecture Parameters
- ✅ Number of attention heads
- ✅ Number of transformer layers
- ✅ Hidden dimension size
- ✅ FFN intermediate size
- ✅ Vocabulary size
- ✅ Maximum sequence length
- ✅ Dropout rates

### Training Parameters
- ✅ Batch size and gradient accumulation
- ✅ Learning rate and warmup
- ✅ Number of epochs
- ✅ Mixed precision (FP16)
- ✅ Gradient checkpointing

## 📖 Documentation

- **GETTING_STARTED.md**: Step-by-step setup and first experiment
- **USAGE.md**: Detailed API documentation and examples
- **README.md**: Package overview and features
- **Examples/**: Working code for common experiments

## 🛠️ Technical Stack

- **PyTorch**: Model implementation and training
- **Transformers**: GPT-2 architecture base
- **Datasets**: HuggingFace dataset integration
- **Matplotlib/Seaborn**: Visualization
- **Pandas**: Results analysis

## ✅ Testing

```bash
# Run verification tests
python test_package.py

# Tests cover:
# - Package imports
# - Configuration creation
# - Model instantiation
# - Data loading
# - Metrics calculation
```

## 🤝 Contributing

This package is designed to be easily extensible:

1. **Add new architectures**: Extend `model.py`
2. **Add new metrics**: Extend `metrics.py`
3. **Add new datasets**: Extend `dataset.py`
4. **Add new visualizations**: Extend `analysis.py`

## 📝 Citation

```bibtex
@software{llm_arch_experiments,
  title={LLM Architecture Experimentation Framework},
  year={2024},
  url={https://github.com/yourusername/llm-arch-experiments}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🎓 Use Cases in Research

This framework is ideal for:

- **Architecture Search**: Systematic exploration of design choices
- **Ablation Studies**: Understanding component contributions
- **Efficiency Research**: Finding optimal size/performance trade-offs
- **Educational Purposes**: Learning about transformer architectures
- **Baseline Comparisons**: Standardized evaluation setup

## 🚀 Next Steps

1. **Clone and Install**: Follow GETTING_STARTED.md
2. **Run Examples**: Try the demo experiments
3. **Customize**: Modify for your research questions
4. **Publish**: Share your findings with the community

## 💬 Support

For issues, questions, or contributions:
- Check the documentation files
- Review example scripts
- Open an issue on GitHub

---

**Ready to start experimenting?** Jump to GETTING_STARTED.md!
