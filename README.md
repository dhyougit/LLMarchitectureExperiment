# 🧪 LLM Architecture Experimentation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-ready Python package for systematic experimentation with transformer architecture modifications. Perfect for LLM researchers exploring how architectural choices impact model performance.

## ✨ Features

- 🔧 Easy modification of architectural parameters (heads, layers, hidden dims, etc.)
- 📊 Comprehensive metrics tracking (loss, perplexity, training time, memory usage)
- 🔄 Experiment versioning and comparison
- 💾 Automatic checkpointing and result logging
- 📈 Visualization tools for comparing experiments

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from experiments.config import ExperimentConfig
from experiments.trainer import ArchitectureExperiment

# Define your baseline configuration
baseline_config = ExperimentConfig(
    model_name="gpt2",
    num_heads=12,
    num_layers=12,
    hidden_size=768,
    experiment_name="baseline"
)

# Run baseline experiment
baseline_exp = ArchitectureExperiment(baseline_config)
baseline_results = baseline_exp.run()

# Test with modified architecture (5 heads)
modified_config = baseline_config.copy()
modified_config.num_heads = 5
modified_config.experiment_name = "five_heads"

modified_exp = ArchitectureExperiment(modified_config)
modified_results = modified_exp.run()

# Compare results
from experiments.analysis import compare_experiments
compare_experiments([baseline_results, modified_results])
```

## Project Structure

```
llm-arch-experiments/
├── experiments/
│   ├── __init__.py
│   ├── config.py          # Configuration classes
│   ├── model.py           # Model architecture definitions
│   ├── trainer.py         # Training and experiment logic
│   ├── metrics.py         # Metrics calculation
│   └── analysis.py        # Results analysis and visualization
├── data/
│   └── dataset.py         # Data loading utilities
├── results/               # Experiment results directory
├── examples/
│   └── run_head_experiment.py
├── requirements.txt
└── README.md
```

## Usage Examples

See `examples/run_head_experiment.py` for a complete example of running attention head experiments.

## Metrics Tracked

- Training loss
- Validation loss
- Perplexity
- Training time per epoch
- GPU memory usage
- Model parameter count
- Inference speed

## License

MIT
