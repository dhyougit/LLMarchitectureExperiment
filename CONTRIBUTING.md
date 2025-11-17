# Contributing to LLM Architecture Experimentation Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

1. **Fork the repository**
2. **Clone your fork**: `git clone <your-fork-url>`
3. **Install in development mode**: `pip install -e .`
4. **Run tests**: `python test_package.py`
5. **Create a branch**: `git checkout -b feature/your-feature`
6. **Make changes and commit**: `git commit -m "Add feature"`
7. **Push and create PR**: `git push origin feature/your-feature`

## 📋 Types of Contributions

### 🐛 Bug Reports
- Use GitHub Issues
- Include minimal reproducible example
- Specify environment (OS, Python version, GPU)
- Include error messages and stack traces

### 💡 Feature Requests
- Describe the use case
- Explain why it's useful
- Provide example usage (pseudo-code is fine)

### 📝 Documentation Improvements
- Fix typos, clarify explanations
- Add examples
- Improve docstrings

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- New architecture support
- New metrics

## 🎯 Development Setup

```bash
# Clone repo
git clone <repo-url>
cd llm-arch-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .
pip install pytest black flake8

# Run tests
python test_package.py
```

## 📐 Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

```python
def calculate_metric(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Calculate metrics for the model.
    
    Args:
        model: PyTorch model to evaluate
        data: Input data tensor
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing metric names and values
    """
    # Implementation
    pass
```

### Code Formatting
```bash
# Format code with black
black experiments/ data/ examples/

# Check with flake8
flake8 experiments/ data/ examples/ --max-line-length=100
```

## 🏗️ Project Structure

```
llm-arch-experiments/
├── experiments/        # Core framework
│   ├── config.py      # Add new config options here
│   ├── model.py       # Add new architectures here
│   ├── trainer.py     # Modify training logic here
│   ├── metrics.py     # Add new metrics here
│   └── analysis.py    # Add new visualizations here
├── data/              # Data loading
│   └── dataset.py     # Add new datasets here
├── examples/          # Example scripts
│   └── *.py          # Add new examples here
└── tests/            # Tests (to be added)
    └── test_*.py     # Add tests here
```

## 🧪 Adding New Features

### Adding a New Architecture

1. Edit `experiments/model.py`:
```python
class CustomArchitectureModel:
    @staticmethod
    def from_config(config_dict: Dict[str, Any]) -> nn.Module:
        """Create model with custom architecture."""
        # Your implementation
        return model
```

2. Update `experiments/config.py` if needed:
```python
@dataclass
class ExperimentConfig:
    # Add new parameters
    new_parameter: int = 42
```

3. Add example in `examples/`:
```python
# examples/run_custom_architecture.py
from experiments import ExperimentConfig, ArchitectureExperiment

config = ExperimentConfig(
    architecture_type="custom",
    new_parameter=100
)
```

4. Update documentation

### Adding a New Metric

1. Edit `experiments/metrics.py`:
```python
@staticmethod
def calculate_custom_metric(model, data) -> float:
    """Calculate custom metric."""
    # Your implementation
    return metric_value
```

2. Update `MetricsCalculator.calculate_metrics()`:
```python
metrics = {
    'loss': avg_loss,
    'perplexity': perplexity,
    'custom_metric': calculate_custom_metric(model, data)
}
```

3. Update `ExperimentResults` in `config.py`:
```python
@dataclass
class ExperimentResults:
    # Add new field
    custom_metric: Optional[float] = None
```

### Adding a New Dataset

1. Edit `data/dataset.py`:
```python
class CustomDataset:
    def __init__(self, ...):
        # Your implementation
        pass
    
    def get_train_dataloader(self):
        # Your implementation
        pass
```

2. Update `get_dataloaders()`:
```python
def get_dataloaders(config):
    if config.dataset_name == "custom":
        dataset = CustomDataset(...)
    # ...
```

## ✅ Testing Guidelines

### Writing Tests
```python
# tests/test_config.py
def test_config_creation():
    config = ExperimentConfig(
        num_heads=8,
        hidden_size=512
    )
    assert config.hidden_size % config.num_heads == 0
```

### Running Tests
```bash
# Run all tests
python test_package.py

# Run specific test
pytest tests/test_config.py
```

## 📝 Documentation Guidelines

### Docstring Format
```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    One-line summary.
    
    Longer description if needed. Explain what the function does,
    any important details, edge cases, etc.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this happens
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### README Updates
- Keep examples up-to-date
- Add new features to feature list
- Update installation if dependencies change

## 🔄 Pull Request Process

1. **Before submitting**:
   - Run tests: `python test_package.py`
   - Format code: `black .`
   - Check style: `flake8 .`
   - Update documentation

2. **PR Description**:
   - Clear title
   - Describe what changed
   - Link related issues
   - Include example usage if applicable

3. **Review Process**:
   - Address reviewer comments
   - Keep PR focused (one feature/fix per PR)
   - Squash commits before merge

## 🎨 Commit Message Guidelines

```
type(scope): short description

Longer description if needed.

- Bullet points for details
- Reference issues: Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples**:
```
feat(metrics): add BLEU score calculation
fix(trainer): handle empty validation set
docs(readme): update installation instructions
```

## 🐛 Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU Memory
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

### Profile Code
```python
import time
start = time.time()
# Your code
print(f"Took {time.time()-start:.2f}s")
```

## 📊 Performance Guidelines

- Keep functions focused (one responsibility)
- Avoid unnecessary loops
- Use vectorized operations
- Profile before optimizing
- Document performance considerations

## 🤝 Code Review Checklist

**For Reviewers**:
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No unnecessary changes
- [ ] Performance considered
- [ ] Error handling appropriate

**For Contributors**:
- [ ] Self-reviewed code
- [ ] Added tests for new features
- [ ] Updated relevant documentation
- [ ] Checked for breaking changes
- [ ] Tested on relevant environments

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open an Issue
- **Feature Ideas**: Open an Issue with `enhancement` label
- **Documentation**: Open an Issue with `documentation` label

## 🎓 Learning Resources

- [PyTorch Docs](https://pytorch.org/docs/)
- [Transformers Docs](https://huggingface.co/docs/transformers/)
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Type Hints Guide](https://docs.python.org/3/library/typing.html)

## 🌟 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing! 🙏
