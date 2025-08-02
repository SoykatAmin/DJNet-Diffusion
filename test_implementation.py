#!/usr/bin/env python3
"""
Quick test script to verify DJNet implementation works
Tests data loading, model creation, and a few training steps
"""

import torch
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import load_config
from src.data.dataset import create_dataloaders
from src.models.diffusion_model import DJNetDiffusionModel

def test_data_loading():
    """Test that data loading works with the actual dataset structure."""
    print("Testing data loading...")
    
    # Load test configs
    training_config = load_config('configs/training_config_cpu_test.yaml')
    model_config = load_config('configs/model_config_cpu_test.yaml')
    config = {**model_config, **training_config}
    
    try:
        # Create dataloaders with small dataset
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        print(f" Created dataloaders:")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        
        # Test loading a single batch
        batch = next(iter(train_loader))
        print(f" Loaded batch with keys: {list(batch.keys())}")
        print(f"   Preceding spec shape: {batch['preceding_spectrogram'].shape}")
        print(f"   Following spec shape: {batch['following_spectrogram'].shape}")
        print(f"   Target spec shape: {batch['target_transition_spectrogram'].shape}")
        print(f"   Transition types: {batch['transition_type']}")
        print(f"   Tempos: {batch['avg_tempo']}")
        
        return config, train_loader
        
    except Exception as e:
        print(f" Data loading failed: {str(e)}")
        raise

def test_model_creation(config):
    """Test that the model can be created and runs forward pass."""
    print("\n Testing model creation...")
    
    try:
        # Create model
        model = DJNetDiffusionModel(config)
        print(f" Created DJNetDiffusionModel")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f" Model creation failed: {str(e)}")
        raise

def test_forward_pass(model, config, train_loader):
    """Test a forward pass through the model."""
    print("\n Testing forward pass...")
    
    try:
        # Get a batch
        batch = next(iter(train_loader))
        
        # Prepare input
        preceding_spec = batch['preceding_spectrogram']
        following_spec = batch['following_spectrogram']  
        target_spec = batch['target_transition_spectrogram']
        tempo = batch['avg_tempo']
        transition_length = batch['transition_length']
        transition_types = batch['transition_type']
        
        # Create model input (concatenate context + target)
        model_input = torch.cat([preceding_spec, following_spec, target_spec], dim=1)
        timesteps = torch.randint(0, 10, (model_input.shape[0],))  # Small timesteps for testing
        
        print(f"   Input shape: {model_input.shape}")
        print(f"   Timesteps: {timesteps}")
        
        # Forward pass
        with torch.no_grad():
            # Test forward pass
            output = model(
                sample=model_input,
                timestep=timesteps,
                tempo=tempo,
                transition_types=transition_types,
                transition_lengths=transition_length,
                return_dict=False
            )
        
        print(f" Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f" Forward pass failed: {str(e)}")
        raise

def test_training_step(config, train_loader):
    """Test a single training step."""
    print("\n Testing training step...")
    
    try:
        from diffusers import DDPMScheduler
        from src.models.diffusion_model import DJNetDiffusionModel
        import torch.nn.functional as F
        
        # Create model and scheduler
        model = DJNetDiffusionModel(config)
        scheduler = DDPMScheduler(
            num_train_timesteps=config['scheduler']['num_train_timesteps'],
            beta_start=config['scheduler']['beta_start'],
            beta_end=config['scheduler']['beta_end'],
            beta_schedule=config['scheduler']['beta_schedule']
        )
        
        # Get batch
        batch = next(iter(train_loader))
        preceding_spec = batch['preceding_spectrogram']
        following_spec = batch['following_spectrogram']
        target_spec = batch['target_transition_spectrogram']
        tempo = batch['avg_tempo']
        transition_length = batch['transition_length']
        transition_types = batch['transition_type']
        
        # Simulate training step
        batch_size = target_spec.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,))
        noise = torch.randn_like(target_spec)
        noisy_target = scheduler.add_noise(target_spec, noise, timesteps)
        
        # Model input
        model_input = torch.cat([preceding_spec, following_spec, noisy_target], dim=1)
        
        # Forward pass
        model_output = model(
            sample=model_input,
            timestep=timesteps,
            tempo=tempo,
            transition_types=transition_types,
            transition_lengths=transition_length,
            return_dict=False
        )
        
        # Calculate loss
        loss = F.mse_loss(model_output, noise)
        
        print(f" Training step successful!")
        print(f"   Loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f" Training step failed: {str(e)}")
        raise

def main():
    print("🧪 DJNet Implementation Test\n")
    print("This will test the implementation with your actual dataset...")
    
    try:
        # Test 1: Data loading
        config, train_loader = test_data_loading()
        
        # Test 2: Model creation  
        model = test_model_creation(config)
        
        # Test 3: Forward pass
        test_forward_pass(model, config, train_loader)
        
        # Test 4: Training step
        test_training_step(config, train_loader)
        
        print("\n All tests passed! The implementation is working correctly.")
        print("\n Next steps:")
        print("1. For CPU training (slow): python scripts/train.py --config configs/training_config_cpu_test.yaml --model_config configs/model_config_cpu_test.yaml")
        print("2. For Colab/Kaggle: Upload this code and use the full configs")
        print("3. The dataset structure has been correctly adapted for your format")
        
    except Exception as e:
        print(f"\n Test failed: {str(e)}")
        print("Please check the error and fix any issues before proceeding.")
        return False
    
    return True

if __name__ == '__main__':
    main()
