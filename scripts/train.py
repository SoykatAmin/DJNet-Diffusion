#!/usr/bin/env python3
"""
Training script for DJNet Diffusion Model
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import DJNetTrainer, load_config


def main():
    parser = argparse.ArgumentParser(description='Train DJNet Diffusion Model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configurations
    training_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Merge configs
    config = {**model_config, **training_config}
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Training configuration: {args.config}")
    print(f"Model configuration: {args.model_config}")
    
    # Initialize trainer
    trainer = DJNetTrainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        print("Checkpoint saved")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
