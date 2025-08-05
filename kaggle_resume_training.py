# Kaggle Resume Training Cell
# Copy this code into your Kaggle notebook to resume training

import torch
import sys
from pathlib import Path

# Add the project to path (adjust if your structure is different)
sys.path.append('/kaggle/working/DJNet-Diffusion/src')

from src.training.trainer import DJNetTrainer, load_config

def resume_training_kaggle():
    """Resume training in Kaggle environment."""
    
    print("🔄 Resuming DJNet Training in Kaggle")
    print("=" * 50)
    
    # Setup paths (adjust these to match your Kaggle setup)
    project_root = Path('/kaggle/working/DJNet-Diffusion')
    checkpoint_dir = project_root / 'checkpoints'
    
    # Load configurations
    model_config = load_config(project_root / 'configs/model_config.yaml')
    training_config = load_config(project_root / 'configs/training_config.yaml') 
    config = {**model_config, **training_config}
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Initialize trainer
    trainer = DJNetTrainer(config, device)
    
    # Auto-resume from latest checkpoint
    if trainer.auto_resume():
        print(f"📊 Resuming from epoch {trainer.current_epoch + 1}")
        print(f"📈 Best validation loss so far: {trainer.best_val_loss:.6f}")
    else:
        print("🆕 No checkpoints found, starting fresh training")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared GPU cache")
    
    # Start/resume training
    try:
        trainer.train()
        print("🎉 Training completed!")
        return trainer
    except Exception as e:
        print(f"❌ Training error: {e}")
        # Save emergency checkpoint
        try:
            trainer.save_checkpoint(trainer.current_epoch, is_best=False)
            print("💾 Emergency checkpoint saved")
        except:
            pass
        return trainer

# Run the resume function
trainer = resume_training_kaggle()

# Optional: Display training progress info
if 'trainer' in locals():
    print(f"\n📊 Final Training Status:")
    print(f"   Completed epochs: {trainer.current_epoch}")
    print(f"   Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"   Model saved in: {trainer.checkpoint_dir}")

# Check available disk space (useful in Kaggle)
import shutil
usage = shutil.disk_usage('/kaggle/working')
print(f"\n💾 Disk Usage:")
print(f"   Total: {usage.total / 1e9:.1f}GB")
print(f"   Used: {usage.used / 1e9:.1f}GB")
print(f"   Free: {usage.free / 1e9:.1f}GB")
