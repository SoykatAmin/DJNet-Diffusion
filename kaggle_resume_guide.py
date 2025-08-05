#!/usr/bin/env python3
"""
Guide for resuming training from Kaggle checkpoints
"""

print("🔄 Resuming Training from Kaggle Checkpoints")
print("=" * 50)

print("\n📋 STEP-BY-STEP GUIDE:")
print("\n1. DOWNLOAD CHECKPOINTS FROM KAGGLE:")
print("   a) In your Kaggle notebook, go to the 'Output' tab")
print("   b) Find your checkpoint files (usually in /kaggle/working/checkpoints/)")
print("   c) Download the checkpoint files you want:")
print("      - latest_model.pth (most recent)")
print("      - best_model.pth (best validation loss)")
print("      - checkpoint_epoch_X.pth (specific epoch)")

print("\n2. TRANSFER TO LOCAL MACHINE:")
print("   a) Place downloaded checkpoints in your local 'checkpoints/' folder")
print("   b) Your local checkpoints folder should look like:")
print("      checkpoints/")
print("      ├── checkpoint_epoch_15.pth  (from Kaggle)")
print("      ├── best_model.pth           (from Kaggle)")
print("      └── latest_model.pth         (from Kaggle)")

print("\n3. RESUME TRAINING LOCALLY:")
print("   a) Use the existing trainer.load_checkpoint() method")
print("   b) Example code:")

code_example = '''
# Load configs
model_config = load_config('configs/model_config.yaml')
training_config = load_config('configs/training_config.yaml')
config = {**model_config, **training_config}

# Initialize trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = DJNetTrainer(config, device)

# Load Kaggle checkpoint
kaggle_checkpoint = 'checkpoints/checkpoint_epoch_15.pth'  # Your Kaggle checkpoint
trainer.load_checkpoint(kaggle_checkpoint)

# Continue training
trainer.train()
'''

print(code_example)

print("\n4. CONTINUE TRAINING IN KAGGLE:")
print("   a) In your Kaggle notebook, use:")

kaggle_resume_code = '''
# Resume from specific checkpoint
checkpoint_path = '/kaggle/working/checkpoints/checkpoint_epoch_15.pth'
trainer.load_checkpoint(checkpoint_path)

# Or auto-resume from latest
trainer.auto_resume()

# Continue training
trainer.train()
'''

print(kaggle_resume_code)

print("\n📁 YOUR CURRENT CHECKPOINTS:")
import os
if os.path.exists('checkpoints'):
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if checkpoints:
        print("   Local checkpoints found:")
        for ckpt in sorted(checkpoints):
            size_mb = os.path.getsize(f'checkpoints/{ckpt}') / (1024 * 1024)
            print(f"   ✓ {ckpt} ({size_mb:.1f}MB)")
    else:
        print("   ❌ No .pth checkpoints found locally")
        print("   💡 Download your Kaggle checkpoints to resume training")
else:
    print("   ❌ No checkpoints folder found")

print(f"\n{'='*50}")
print("🎯 QUICK ACTIONS:")
print("1. If you have Kaggle checkpoints → Download and place in checkpoints/")
print("2. If you want to continue in Kaggle → Use trainer.auto_resume()")
print("3. If you want local training → Download checkpoints first")
