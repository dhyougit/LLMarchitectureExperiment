"""
Training and experiment orchestration.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import get_scheduler
from tqdm.auto import tqdm
import time
import random
import numpy as np
from pathlib import Path

from .config import ExperimentConfig, ExperimentResults
from .model import CustomGPT2Model
from .metrics import MetricsCalculator, MetricsTracker
from data.dataset import get_dataloaders


class ArchitectureExperiment:
    """Main class for running architecture experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        self._set_seed(config.seed)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloaders = None
        self.metrics_tracker = MetricsTracker()
        
        print(f"Experiment: {config.experiment_name}")
        print(f"Device: {self.device}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def setup(self):
        """Setup model, optimizer, and data."""
        print("\n" + "="*60)
        print("Setting up experiment...")
        print("="*60)
        
        # Create model with custom architecture
        model_config_dict = self.config.get_model_config_dict()
        self.model = CustomGPT2Model.from_config(model_config_dict)
        self.model.to(self.device)
        
        # Print model summary
        CustomGPT2Model.print_model_summary(self.model)
        
        # Setup data
        print("\nLoading data...")
        self.dataloaders = get_dataloaders(self.config)
        
        # Calculate total training steps
        num_training_steps = len(self.dataloaders['train']) * self.config.num_epochs
        num_training_steps = num_training_steps // self.config.gradient_accumulation_steps
        
        if self.config.max_steps is not None:
            num_training_steps = min(num_training_steps, self.config.max_steps)
        
        print(f"Total training steps: {num_training_steps}")
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_fp16 else None
        
        print("Setup complete!\n")
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Early stopping if max_steps reached
            if self.config.max_steps and step >= self.config.max_steps:
                break
        
        avg_loss = total_loss / num_batches
        perplexity = MetricsCalculator.calculate_perplexity(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def evaluate(self) -> dict:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating...")
        metrics = MetricsCalculator.calculate_metrics(
            self.model,
            self.dataloaders['val'],
            device=str(self.device)
        )
        return metrics
    
    def run(self) -> ExperimentResults:
        """
        Run the full experiment.
        
        Returns:
            ExperimentResults object with all metrics
        """
        # Setup
        self.setup()
        
        # Save configuration
        self.config.save()
        
        # Reset memory tracking
        MetricsCalculator.reset_gpu_memory_stats()
        
        # Training loop
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")
        
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Track training metrics
            self.metrics_tracker.add_train_metrics(
                loss=train_metrics['loss'],
                perplexity=train_metrics['perplexity'],
                learning_rate=self.scheduler.get_last_lr()[0],
                epoch=epoch,
                step=epoch
            )
            
            # Evaluate
            val_metrics = self.evaluate()
            self.metrics_tracker.add_val_metrics(
                loss=val_metrics['loss'],
                perplexity=val_metrics['perplexity']
            )
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            print()
        
        total_time = time.time() - start_time
        
        # Get final metrics
        final_val_metrics = self.evaluate()
        peak_memory = MetricsCalculator.get_gpu_memory_usage()
        param_counts = CustomGPT2Model.count_parameters(self.model)
        
        # Create results object
        all_metrics = self.metrics_tracker.get_all_metrics()
        results = ExperimentResults(
            experiment_name=self.config.experiment_name,
            config=self.config,
            train_loss_history=all_metrics['train_loss'],
            val_loss_history=all_metrics['val_loss'],
            train_perplexity_history=all_metrics['train_perplexity'],
            val_perplexity_history=all_metrics['val_perplexity'],
            final_train_loss=all_metrics['train_loss'][-1],
            final_val_loss=final_val_metrics['loss'],
            final_train_perplexity=all_metrics['train_perplexity'][-1],
            final_val_perplexity=final_val_metrics['perplexity'],
            total_training_time=total_time,
            avg_epoch_time=np.mean(epoch_times),
            peak_memory_mb=peak_memory,
            total_params=param_counts['total'],
            trainable_params=param_counts['trainable'],
            tokens_per_second=final_val_metrics['tokens_per_second']
        )
        
        # Save results
        results.save()
        
        # Save model checkpoint
        model_path = self.config.experiment_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        print("="*60)
        print("Experiment Complete!")
        print("="*60)
        print(f"Results saved to: {self.config.experiment_dir}")
        print(f"Final validation perplexity: {results.final_val_perplexity:.2f}")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Peak memory usage: {peak_memory:.2f}MB")
        
        return results
