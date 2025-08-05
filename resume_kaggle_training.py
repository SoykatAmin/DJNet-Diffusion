#!/usr/bin/env python3
"""
Resume training from your Kaggle-trained DJNet model
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.trainer import DJNetTrainer, load_config

def resume_from_kaggle():
    """Resume training from Kaggle checkpoint."""
    
    print("🔄 Resuming DJNet Training from Kaggle")
    print("=" * 50)
    
    # Load configurations
    try:
        model_config = load_config('configs/model_config.yaml')
        training_config = load_config('configs/training_config.yaml')
        config = {**model_config, **training_config}
        print("✓ Loaded configurations")
    except Exception as e:
        print(f"❌ Error loading configs: {e}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize trainer
    trainer = DJNetTrainer(config, device)
    
    # Check available checkpoints
    checkpoint_dir = Path('checkpoints')
    available_checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not available_checkpoints:
        print("❌ No checkpoints found!")
        print("💡 Download your Kaggle checkpoints to the 'checkpoints/' folder first")
        return
    
    print(f"\n📁 Available checkpoints:")
    for i, ckpt in enumerate(available_checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"   {i+1}. {ckpt.name} ({size_mb:.1f}MB)")
    
    # Try to automatically resume from the best checkpoint
    best_checkpoints = [ckpt for ckpt in available_checkpoints if 'best' in ckpt.name]
    latest_checkpoints = [ckpt for ckpt in available_checkpoints if 'latest' in ckpt.name]
    epoch_checkpoints = [ckpt for ckpt in available_checkpoints if 'epoch' in ckpt.name]
    
    # Prioritize: best > latest > highest epoch
    checkpoint_to_load = None
    
    if best_checkpoints:
        # Use the most recent best checkpoint
        checkpoint_to_load = max(best_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"\n🏆 Using best checkpoint: {checkpoint_to_load.name}")
    elif latest_checkpoints:
        checkpoint_to_load = max(latest_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"\n📈 Using latest checkpoint: {checkpoint_to_load.name}")
    elif epoch_checkpoints:
        # Find highest epoch number
        highest_epoch_ckpt = max(epoch_checkpoints, 
                               key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
        checkpoint_to_load = highest_epoch_ckpt
        print(f"\n📊 Using highest epoch checkpoint: {checkpoint_to_load.name}")
    else:
        # Use the most recently modified checkpoint
        checkpoint_to_load = max(available_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"\n⏰ Using most recent checkpoint: {checkpoint_to_load.name}")
    
    # Load the selected checkpoint
    try:
        trainer.load_checkpoint(str(checkpoint_to_load))
        
        print(f"\n📊 Training Status:")
        print(f"   ✓ Loaded from epoch: {trainer.current_epoch}")
        print(f"   ✓ Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"   ✓ Target epochs: {config['training']['num_epochs']}")
        
        remaining_epochs = config['training']['num_epochs'] - trainer.current_epoch
        print(f"   📈 Epochs remaining: {remaining_epochs}")
        
        if remaining_epochs <= 0:
            print(f"\n🎉 Training already completed!")
            print(f"   Current epoch ({trainer.current_epoch}) >= target epochs ({config['training']['num_epochs']})")
            
            # Ask if user wants to extend training
            response = input("\n🤔 Extend training? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                new_epochs = int(input("How many additional epochs? "))
                config['training']['num_epochs'] = trainer.current_epoch + new_epochs
                print(f"✓ Extended training to {config['training']['num_epochs']} epochs")
            else:
                print("Training not extended. Exiting.")
                return
        
        # Continue training
        print(f"\n🚀 Resuming training from epoch {trainer.current_epoch + 1}")
        print("Press Ctrl+C to stop training gracefully")
        
        try:
            trainer.train()
            print("\n🎉 Training completed successfully!")
        except KeyboardInterrupt:
            print(f"\n⏸️  Training interrupted by user at epoch {trainer.current_epoch}")
            print("Checkpoints saved. You can resume later using this script.")
        except Exception as e:
            print(f"\n❌ Training error: {str(e)}")
            raise e
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {str(e)}")
        print("💡 Make sure the checkpoint is compatible with your current model configuration")
        return

if __name__ == "__main__":
    resume_from_kaggle()
