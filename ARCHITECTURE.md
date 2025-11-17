# Framework Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM Arch Experiments                       │
│                     (User Interface)                         │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  ExperimentConfig                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Architecture Parameters (heads, layers, etc.)      │   │
│  │ - Training Parameters (lr, epochs, etc.)            │   │
│  │ - Data Parameters (dataset, size, etc.)             │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│               ArchitectureExperiment                         │
│                 (Main Orchestrator)                          │
└─────┬──────────┬──────────┬──────────┬────────────┬─────────┘
      │          │          │          │            │
      ▼          ▼          ▼          ▼            ▼
┌──────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│  Model   │ │  Data  │ │  Metrics │ │ Trainer │ │ Analysis │
│ Builder  │ │ Loader │ │Calculator│ │  Loop   │ │  Tools   │
└──────────┘ └────────┘ └──────────┘ └─────────┘ └──────────┘
      │          │          │          │            │
      ▼          ▼          ▼          ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ExperimentResults                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Training History (loss, perplexity)               │   │
│  │ - Performance Metrics (time, memory)                │   │
│  │ - Model Checkpoints                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Comparison & Visualization                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Side-by-side comparisons                          │   │
│  │ - Training curves                                   │   │
│  │ - Summary reports                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ExperimentConfig
**Purpose**: Centralized configuration management

**Key Responsibilities**:
- Validate architecture parameters (e.g., hidden_size divisible by num_heads)
- Provide sensible defaults
- Support configuration serialization (save/load)
- Create experiment directories

**Key Methods**:
- `copy()`: Create configuration variants
- `save()`: Persist to JSON
- `get_model_config_dict()`: Convert to model parameters

### 2. CustomGPT2Model
**Purpose**: Create GPT-2 models with custom architectures

**Key Responsibilities**:
- Build models from configuration
- Count parameters
- Provide model information

**Key Methods**:
- `from_config()`: Create model from config dict
- `count_parameters()`: Get parameter counts
- `get_model_info()`: Get detailed architecture info

### 3. LanguageModelDataset
**Purpose**: Handle data loading and preprocessing

**Key Responsibilities**:
- Load datasets from HuggingFace
- Tokenize text
- Create PyTorch DataLoaders
- Support train/validation splits

**Key Methods**:
- `get_train_dataloader()`: Get training data
- `get_val_dataloader()`: Get validation data

### 4. ArchitectureExperiment
**Purpose**: Main orchestrator for experiments

**Key Responsibilities**:
- Set up model, data, and optimizer
- Run training loop
- Track metrics
- Save results

**Key Methods**:
- `setup()`: Initialize all components
- `train_epoch()`: Train for one epoch
- `evaluate()`: Evaluate model
- `run()`: Execute full experiment

### 5. MetricsCalculator
**Purpose**: Calculate and track metrics

**Key Responsibilities**:
- Compute loss and perplexity
- Measure training time
- Monitor memory usage
- Track inference speed

**Key Methods**:
- `calculate_perplexity()`: Convert loss to perplexity
- `calculate_metrics()`: Evaluate on dataset
- `get_gpu_memory_usage()`: Get memory stats

### 6. Analysis Tools
**Purpose**: Compare and visualize experiments

**Key Responsibilities**:
- Generate comparison plots
- Create summary tables
- Produce detailed reports

**Key Functions**:
- `compare_experiments()`: Side-by-side comparison
- `plot_training_curves()`: Visualize training
- `generate_summary_report()`: Text summary

## Data Flow

```
User Input (Config)
        │
        ▼
    Setup Phase
    ├─ Create Model
    ├─ Load Data
    ├─ Initialize Optimizer
    └─ Set up Metrics Tracker
        │
        ▼
   Training Loop (each epoch)
    ├─ Forward pass
    ├─ Backward pass
    ├─ Update weights
    ├─ Track metrics
    └─ Evaluate
        │
        ▼
   Results Collection
    ├─ Final metrics
    ├─ Training history
    ├─ Model checkpoint
    └─ Performance stats
        │
        ▼
    Save Results
    ├─ config.json
    ├─ results.json
    └─ model.pt
        │
        ▼
   Analysis (optional)
    ├─ Load results
    ├─ Compare experiments
    └─ Generate visualizations
```

## File Organization

```
experiments/
├── __init__.py          # Package exports
├── config.py            # Configuration classes
│   ├── ExperimentConfig
│   └── ExperimentResults
├── model.py             # Model creation
│   └── CustomGPT2Model
├── trainer.py           # Training orchestration
│   └── ArchitectureExperiment
├── metrics.py           # Metrics calculation
│   ├── MetricsCalculator
│   └── MetricsTracker
└── analysis.py          # Visualization
    ├── compare_experiments()
    ├── plot_training_curves()
    └── generate_summary_report()

data/
├── __init__.py
└── dataset.py           # Data loading
    └── LanguageModelDataset

examples/
├── run_head_experiment.py    # Attention head experiments
└── run_layer_experiment.py   # Layer depth experiments
```

## Execution Flow Example

```python
# 1. User creates config
config = ExperimentConfig(
    experiment_name="test",
    num_heads=5,
    num_layers=6
)

# 2. Initialize experiment
experiment = ArchitectureExperiment(config)

# 3. Setup phase
experiment.setup()
    ├─ model = CustomGPT2Model.from_config(config_dict)
    ├─ dataloaders = get_dataloaders(config)
    ├─ optimizer = AdamW(model.parameters())
    └─ scheduler = get_linear_schedule_with_warmup()

# 4. Training loop
for epoch in range(num_epochs):
    train_metrics = experiment.train_epoch(epoch)
    val_metrics = experiment.evaluate()
    tracker.add_metrics(train_metrics, val_metrics)

# 5. Save results
results = ExperimentResults(
    config=config,
    metrics=tracker.get_all_metrics(),
    performance_stats=performance_stats
)
results.save()

# 6. Analysis (optional)
compare_experiments([results1, results2, results3])
```

## Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Configuration-driven**: All experiments defined by config objects
3. **Automatic tracking**: Metrics logged without manual intervention
4. **Reproducibility**: Seeds and configs ensure repeatability
5. **Extensibility**: Easy to add new models, metrics, or analyses

## Extension Points

### Adding New Model Architecture
```python
# In model.py
class CustomTransformerModel:
    @staticmethod
    def from_config(config_dict):
        # Your implementation
        pass
```

### Adding New Metrics
```python
# In metrics.py
@staticmethod
def calculate_custom_metric(model, data):
    # Your implementation
    pass
```

### Adding New Dataset
```python
# In dataset.py
class CustomDataset:
    def __init__(self, ...):
        # Your implementation
        pass
```

## Performance Considerations

1. **Memory Optimization**:
   - Mixed precision training (FP16)
   - Gradient checkpointing
   - Batch size tuning

2. **Speed Optimization**:
   - DataLoader workers
   - GPU utilization
   - Gradient accumulation

3. **Storage Optimization**:
   - JSON for configs/results
   - PyTorch checkpoints for models
   - Selective metric logging
