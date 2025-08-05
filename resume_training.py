#!/usr/bin/env python3
"""
Resume training script for DJNet-Diffusion
"""

import torch
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.trainer import DJNetTrainer, load_config

def main():
    parser = argparse.ArgumentParser(description='Resume DJNet training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (if not provided, uses latest)')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Automatically find and resume from latest checkpoint')
    parser.add_argument('--list-checkpoints', action='store_true',
                       help='List available checkpoints and exit')
    
    args = parser.parse_args()
    
    print("🔄 DJNet Training Resume")
    print("=" * 40)
    
    # List checkpoints if requested
    if args.list_checkpoints:
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('*.pth'))
            print(f"Available checkpoints in {checkpoint_dir}:")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  📁 {ckpt.name} ({size_mb:.1f}MB)")
        else:
            print("❌ No checkpoints directory found")
        return
    
    # Load configs
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    config = {**model_config, **training_config}
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize trainer
    trainer = DJNetTrainer(config, device)
    
    # Determine which checkpoint to load
    if args.checkpoint:
        # Use specific checkpoint
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"📂 Loading specific checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        
    elif args.auto_resume:
        # Auto-resume from latest
        if not trainer.auto_resume():
            print("❌ No checkpoints found for auto-resume")
            return
            
    else:
        # Interactive selection
        checkpoint_dir = Path('checkpoints')
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if not checkpoints:
            print("❌ No epoch checkpoints found")
            print("Available files:")
            for f in checkpoint_dir.glob('*.pth'):
                print(f"  - {f.name}")
            return
        
        print("Available epoch checkpoints:")
        for i, ckpt in enumerate(checkpoints):
            epoch = int(ckpt.stem.split('_')[-1])
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  {i+1}. Epoch {epoch} - {ckpt.name} ({size_mb:.1f}MB)")
        
        try:
            choice = int(input("Select checkpoint (number): ")) - 1
            if 0 <= choice < len(checkpoints):
                checkpoint_path = str(checkpoints[choice])
                trainer.load_checkpoint(checkpoint_path)
            else:
                print("❌ Invalid selection")
                return
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled")
            return
    
    # Show training progress
    print(f"\n📊 Training Status:")
    print(f"   Current epoch: {trainer.current_epoch}")
    print(f"   Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"   Total epochs planned: {config['training']['num_epochs']}")
    print(f"   Epochs remaining: {config['training']['num_epochs'] - trainer.current_epoch}")
    
    # Continue training
    print(f"\n🚀 Resuming training from epoch {trainer.current_epoch + 1}")
    
    try:
        trainer.train()
        print("🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
        print(f"   Last completed epoch: {trainer.current_epoch}")
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
