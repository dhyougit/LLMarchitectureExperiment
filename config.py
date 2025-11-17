"""
Configuration classes for architecture experiments.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single architecture experiment."""
    
    # Experiment identification
    experiment_name: str = "default_experiment"
    base_model: str = "gpt2"  # Can be "gpt2", "gpt2-medium", etc.
    
    # Architecture parameters
    num_heads: int = 12
    num_layers: int = 12
    hidden_size: int = 768
    intermediate_size: Optional[int] = None  # FFN intermediate size
    max_position_embeddings: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    max_steps: Optional[int] = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data parameters
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_seq_length: int = 512
    train_size: Optional[int] = None  # None = use all
    val_size: Optional[int] = None
    
    # Optimization
    use_fp16: bool = True
    gradient_checkpointing: bool = False
    
    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "./results"
    
    # Reproducibility
    seed: int = 42
    
    # Hardware
    device: str = "cuda"  # "cuda" or "cpu"
    
    def __post_init__(self):
        """Set derived values after initialization."""
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        
        # Ensure hidden_size is divisible by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        # Create output directory for this experiment
        self.experiment_dir = Path(self.output_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def copy(self) -> 'ExperimentConfig':
        """Create a deep copy of this configuration."""
        return ExperimentConfig(**asdict(self))
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = self.experiment_dir / "config.json"
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get dictionary of model architecture parameters."""
        return {
            'n_head': self.num_heads,
            'n_layer': self.num_layers,
            'n_embd': self.hidden_size,
            'n_inner': self.intermediate_size,
            'n_positions': self.max_position_embeddings,
            'vocab_size': self.vocab_size,
            'attn_pdrop': self.attention_dropout,
            'embd_pdrop': self.dropout,
            'resid_pdrop': self.dropout,
        }


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    
    experiment_name: str
    config: ExperimentConfig
    
    # Training metrics
    train_loss_history: list = field(default_factory=list)
    val_loss_history: list = field(default_factory=list)
    train_perplexity_history: list = field(default_factory=list)
    val_perplexity_history: list = field(default_factory=list)
    
    # Final metrics
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_train_perplexity: Optional[float] = None
    final_val_perplexity: Optional[float] = None
    
    # Performance metrics
    total_training_time: Optional[float] = None
    avg_epoch_time: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    
    # Inference metrics
    tokens_per_second: Optional[float] = None
    
    def save(self, path: Optional[Path] = None):
        """Save results to JSON file."""
        if path is None:
            path = self.config.experiment_dir / "results.json"
        
        results_dict = {
            'experiment_name': self.experiment_name,
            'config': asdict(self.config),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'train_perplexity_history': self.train_perplexity_history,
            'val_perplexity_history': self.val_perplexity_history,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'final_train_perplexity': self.final_train_perplexity,
            'final_val_perplexity': self.final_val_perplexity,
            'total_training_time': self.total_training_time,
            'avg_epoch_time': self.avg_epoch_time,
            'peak_memory_mb': self.peak_memory_mb,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'tokens_per_second': self.tokens_per_second,
        }
        
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentResults':
        """Load results from JSON file."""
        with open(path, 'r') as f:
            results_dict = json.load(f)
        
        config = ExperimentConfig(**results_dict.pop('config'))
        return cls(config=config, **results_dict)
