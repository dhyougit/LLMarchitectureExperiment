"""
Data loading utilities for language model experiments.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import torch


class LanguageModelDataset:
    """Dataset handler for language modeling experiments."""
    
    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_name: Name of the dataset from HuggingFace
            dataset_config: Configuration of the dataset
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            train_size: Number of training examples (None = all)
            val_size: Number of validation examples (None = all)
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}/{dataset_config}")
        self.dataset = load_dataset(dataset_name, dataset_config)
        
        # Subset if requested
        if train_size is not None and 'train' in self.dataset:
            self.dataset['train'] = self.dataset['train'].select(range(min(train_size, len(self.dataset['train']))))
        
        if val_size is not None and 'validation' in self.dataset:
            self.dataset['validation'] = self.dataset['validation'].select(range(min(val_size, len(self.dataset['validation']))))
        
        # Tokenize dataset
        print("Tokenizing dataset...")
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names if 'train' in self.dataset else [],
            desc="Tokenizing"
        )
        
        print(f"Dataset prepared. Train size: {len(self.tokenized_dataset.get('train', []))}, "
              f"Val size: {len(self.tokenized_dataset.get('validation', []))}")
    
    def _tokenize_function(self, examples):
        """Tokenize examples."""
        # Tokenize the texts
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def get_train_dataloader(self, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        if 'train' not in self.tokenized_dataset:
            raise ValueError("Training split not available in dataset")
        
        return DataLoader(
            self.tokenized_dataset['train'],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def get_val_dataloader(self, batch_size: int = 8) -> DataLoader:
        """Get validation dataloader."""
        split_name = 'validation' if 'validation' in self.tokenized_dataset else 'test'
        
        if split_name not in self.tokenized_dataset:
            raise ValueError(f"Validation/test split not available in dataset")
        
        return DataLoader(
            self.tokenized_dataset[split_name],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Collate function for dataloader."""
        # Convert list of dicts to dict of lists
        batch_dict = {
            key: torch.tensor([example[key] for example in batch])
            for key in batch[0].keys()
        }
        return batch_dict


def get_dataloaders(
    config: 'ExperimentConfig'
) -> Dict[str, DataLoader]:
    """
    Create dataloaders from experiment configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    dataset = LanguageModelDataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        tokenizer_name=config.base_model,
        max_length=config.max_seq_length,
        train_size=config.train_size,
        val_size=config.val_size,
    )
    
    train_dataloader = dataset.get_train_dataloader(
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_dataloader = dataset.get_val_dataloader(
        batch_size=config.batch_size
    )
    
    return {
        'train': train_dataloader,
        'val': val_dataloader
    }
