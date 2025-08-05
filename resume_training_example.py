#!/usr/bin/env python3
"""
Example script showing how to resume training from a checkpoint
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.trainer import DJNetTrainer, load_config

def resume_training_example():
    """Example of how to resume training from a checkpoint."""
    
    print("🔄 DJNet Training Resume Example")
    print("=" * 50)
    
    # Load configs
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    
    # Merge configs
    config = {**model_config, **training_config}
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = DJNetTrainer(config, device)
    
    # Resume from checkpoint
    checkpoint_path = 'checkpoints/checkpoint_epoch_15.pth'  # Your last checkpoint
    
    if Path(checkpoint_path).exists():
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        print(f"✅ Resumed from epoch {trainer.current_epoch}")
        print(f"   Best validation loss so far: {trainer.best_val_loss:.4f}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth')):
                print(f"   - {ckpt.name}")
        return
    
    # Continue training from the loaded checkpoint
    print(f"🚀 Resuming training from epoch {trainer.current_epoch + 1}")
    trainer.train()

if __name__ == "__main__":
    resume_training_example()
