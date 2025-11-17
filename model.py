"""
Custom model architectures for experimentation.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any


class CustomGPT2Model:
    """Factory for creating GPT2 models with custom architectures."""
    
    @staticmethod
    def from_config(config_dict: Dict[str, Any]) -> GPT2LMHeadModel:
        """
        Create a GPT2 model with custom architecture parameters.
        
        Args:
            config_dict: Dictionary containing architecture parameters
            
        Returns:
            GPT2LMHeadModel with custom configuration
        """
        # Create GPT2 configuration
        model_config = GPT2Config(
            vocab_size=config_dict.get('vocab_size', 50257),
            n_positions=config_dict.get('n_positions', 1024),
            n_embd=config_dict.get('n_embd', 768),
            n_layer=config_dict.get('n_layer', 12),
            n_head=config_dict.get('n_head', 12),
            n_inner=config_dict.get('n_inner', None),
            activation_function=config_dict.get('activation_function', 'gelu_new'),
            resid_pdrop=config_dict.get('resid_pdrop', 0.1),
            embd_pdrop=config_dict.get('embd_pdrop', 0.1),
            attn_pdrop=config_dict.get('attn_pdrop', 0.1),
            layer_norm_epsilon=config_dict.get('layer_norm_epsilon', 1e-5),
            initializer_range=config_dict.get('initializer_range', 0.02),
            use_cache=config_dict.get('use_cache', True),
        )
        
        # Initialize model from scratch with custom config
        model = GPT2LMHeadModel(model_config)
        
        return model
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """
        Count total and trainable parameters in the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    @staticmethod
    def get_model_info(model: GPT2LMHeadModel) -> Dict[str, Any]:
        """
        Get detailed information about the model architecture.
        
        Args:
            model: GPT2 model
            
        Returns:
            Dictionary with model information
        """
        config = model.config
        param_counts = CustomGPT2Model.count_parameters(model)
        
        return {
            'num_layers': config.n_layer,
            'num_heads': config.n_head,
            'hidden_size': config.n_embd,
            'intermediate_size': config.n_inner,
            'vocab_size': config.vocab_size,
            'max_position_embeddings': config.n_positions,
            'head_dim': config.n_embd // config.n_head,
            'total_params': param_counts['total'],
            'trainable_params': param_counts['trainable'],
            'params_millions': param_counts['total'] / 1e6,
        }
    
    @staticmethod
    def print_model_summary(model: GPT2LMHeadModel):
        """Print a summary of the model architecture."""
        info = CustomGPT2Model.get_model_info(model)
        
        print("=" * 60)
        print("Model Architecture Summary")
        print("=" * 60)
        print(f"Number of Layers:          {info['num_layers']}")
        print(f"Number of Attention Heads: {info['num_heads']}")
        print(f"Hidden Size:               {info['hidden_size']}")
        print(f"Intermediate Size (FFN):   {info['intermediate_size']}")
        print(f"Head Dimension:            {info['head_dim']}")
        print(f"Vocabulary Size:           {info['vocab_size']}")
        print(f"Max Sequence Length:       {info['max_position_embeddings']}")
        print("-" * 60)
        print(f"Total Parameters:          {info['total_params']:,}")
        print(f"Trainable Parameters:      {info['trainable_params']:,}")
        print(f"Parameters (millions):     {info['params_millions']:.2f}M")
        print("=" * 60)


def load_pretrained_with_modifications(
    base_model: str,
    modifications: Dict[str, Any]
) -> GPT2LMHeadModel:
    """
    Load a pretrained model and modify its architecture.
    
    Note: This reinitializes the model with new architecture, so pretrained
    weights are not preserved when architecture dimensions change.
    
    Args:
        base_model: Name of the base model (e.g., "gpt2")
        modifications: Dictionary of architecture parameters to modify
        
    Returns:
        Modified GPT2 model
    """
    # Load base configuration
    base_config = GPT2Config.from_pretrained(base_model)
    
    # Apply modifications
    for key, value in modifications.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
    
    # Create model with modified config
    model = GPT2LMHeadModel(base_config)
    
    return model
