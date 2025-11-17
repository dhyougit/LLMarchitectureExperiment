"""
Metrics calculation for language model experiments.
"""
import torch
import numpy as np
from typing import Dict, Any
import time


class MetricsCalculator:
    """Calculate and track metrics during training."""
    
    @staticmethod
    def calculate_perplexity(loss: float) -> float:
        """
        Calculate perplexity from loss.
        
        Args:
            loss: Cross-entropy loss value
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)
    
    @staticmethod
    def calculate_metrics(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Calculate metrics on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Dataloader with evaluation data
            device: Device to run evaluation on
            
        Returns:
            Dictionary with metrics
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()
                num_batches += 1
        
        eval_time = time.time() - start_time
        
        avg_loss = total_loss / num_batches
        perplexity = MetricsCalculator.calculate_perplexity(avg_loss)
        tokens_per_second = total_tokens / eval_time if eval_time > 0 else 0
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_per_second': tokens_per_second,
            'total_tokens': total_tokens,
            'eval_time': eval_time
        }
    
    @staticmethod
    def get_gpu_memory_usage() -> float:
        """
        Get current GPU memory usage in MB.
        
        Returns:
            Memory usage in MB, or 0 if CUDA not available
        """
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    @staticmethod
    def reset_gpu_memory_stats():
        """Reset GPU memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


class MetricsTracker:
    """Track metrics over time during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': [],
            'epoch': [],
            'step': [],
        }
    
    def add_train_metrics(
        self,
        loss: float,
        perplexity: float,
        learning_rate: float,
        epoch: int,
        step: int
    ):
        """Add training metrics."""
        self.metrics['train_loss'].append(loss)
        self.metrics['train_perplexity'].append(perplexity)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['epoch'].append(epoch)
        self.metrics['step'].append(step)
    
    def add_val_metrics(
        self,
        loss: float,
        perplexity: float
    ):
        """Add validation metrics."""
        self.metrics['val_loss'].append(loss)
        self.metrics['val_perplexity'].append(perplexity)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics."""
        return {
            key: values[-1] if values else None
            for key, values in self.metrics.items()
        }
    
    def get_all_metrics(self) -> Dict[str, list]:
        """Get all tracked metrics."""
        return self.metrics.copy()
    
    def print_latest(self):
        """Print the latest metrics."""
        latest = self.get_latest_metrics()
        if latest['train_loss'] is not None:
            print(f"Step {latest['step']}, Epoch {latest['epoch']}: "
                  f"Train Loss={latest['train_loss']:.4f}, "
                  f"Train Perplexity={latest['train_perplexity']:.2f}")
        if latest['val_loss'] is not None:
            print(f"Validation Loss={latest['val_loss']:.4f}, "
                  f"Validation Perplexity={latest['val_perplexity']:.2f}")
