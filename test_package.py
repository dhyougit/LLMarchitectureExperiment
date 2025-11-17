"""
Test script to verify the package installation and basic functionality
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from experiments import (
            ExperimentConfig,
            ExperimentResults,
            CustomGPT2Model,
            ArchitectureExperiment,
            compare_experiments
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test configuration creation."""
    print("\nTesting configuration creation...")
    try:
        from experiments import ExperimentConfig
        
        config = ExperimentConfig(
            experiment_name="test_experiment",
            num_heads=8,
            num_layers=6,
            hidden_size=512,
            batch_size=2,
            num_epochs=1,
            train_size=100,
            val_size=50,
            output_dir="./test_results"
        )
        
        # Test that hidden_size is divisible by num_heads
        assert config.hidden_size % config.num_heads == 0, "Hidden size must be divisible by num_heads"
        
        # Test saving and loading
        config.save()
        loaded_config = ExperimentConfig.load(config.experiment_dir / "config.json")
        assert loaded_config.experiment_name == config.experiment_name
        
        print("✓ Configuration creation and saving successful")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        from experiments import CustomGPT2Model
        
        config_dict = {
            'n_head': 4,
            'n_layer': 2,
            'n_embd': 256,
            'n_inner': 1024,
            'vocab_size': 50257,
            'n_positions': 512,
        }
        
        model = CustomGPT2Model.from_config(config_dict)
        
        # Get model info
        info = CustomGPT2Model.get_model_info(model)
        assert info['num_heads'] == 4
        assert info['num_layers'] == 2
        assert info['hidden_size'] == 256
        
        print(f"✓ Model creation successful ({info['params_millions']:.2f}M parameters)")
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    try:
        from data import LanguageModelDataset
        
        # This will download the dataset if not already cached
        print("  Loading dataset (this may take a moment)...")
        dataset = LanguageModelDataset(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            tokenizer_name="gpt2",
            max_length=128,
            train_size=50,
            val_size=20,
        )
        
        train_loader = dataset.get_train_dataloader(batch_size=2)
        val_loader = dataset.get_val_dataloader(batch_size=2)
        
        # Test that we can get a batch
        batch = next(iter(train_loader))
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        
        print("✓ Data loading successful")
        return True
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        print("  Note: This requires internet connection to download datasets")
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics...")
    try:
        from experiments import MetricsCalculator
        import numpy as np
        
        # Test perplexity calculation
        loss = 2.5
        perplexity = MetricsCalculator.calculate_perplexity(loss)
        expected = np.exp(loss)
        assert abs(perplexity - expected) < 0.001
        
        print("✓ Metrics calculation successful")
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING PACKAGE TESTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_creation),
        ("Model Creation", test_model_creation),
        ("Metrics", test_metrics),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The package is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
