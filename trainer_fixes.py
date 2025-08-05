"""
Fixed trainer with proper device handling and robust checkpoint saving
"""

# Add these fixes to your trainer.py

def save_sample_audio_fixed(self, epoch: int):
    """Generate and save sample audio with proper device handling."""
    try:
        # Get a validation batch
        val_batch = next(iter(self.val_loader))
        
        # Generate samples
        for i in range(min(self.config['logging']['num_audio_samples'], len(val_batch['transition_id']))):
            # Create single sample batch with proper device handling
            sample_batch = {}
            for k, v in val_batch.items():
                if isinstance(v, torch.Tensor):
                    # Ensure tensor is on correct device
                    sample_batch[k] = v[i:i+1].to(self.device)
                else:
                    sample_batch[k] = [v[i]]
            
            # Generate transition
            generated_spec = self.generate_sample(sample_batch)
            
            # Move generated spec to CPU for audio conversion
            generated_spec_cpu = generated_spec.cpu()
            
            # Convert to audio
            from src.utils.audio_utils import spectrogram_to_audio
            generated_audio = spectrogram_to_audio(generated_spec_cpu, self.config)
            
            # Save audio
            save_path = self.checkpoint_dir / f'samples/epoch_{epoch}_sample_{i}.wav'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Ensure audio is on CPU for saving
            if isinstance(generated_audio, torch.Tensor):
                generated_audio = generated_audio.cpu()
            
            import torchaudio
            torchaudio.save(
                str(save_path), 
                generated_audio.unsqueeze(0) if generated_audio.dim() == 1 else generated_audio,
                self.config['audio']['sample_rate']
            )
            
            print(f"Saved sample audio: {save_path}")
            
    except Exception as e:
        print(f"Error saving sample audio: {str(e)}")
        import traceback
        traceback.print_exc()


def save_checkpoint_robust(self, epoch: int, is_best: bool = False):
    """Save checkpoint with robust error handling and disk space management."""
    try:
        # First, check available disk space (rough estimate)
        import shutil
        free_space = shutil.disk_usage(self.checkpoint_dir).free
        estimated_checkpoint_size = 1.5e9  # ~1.5GB estimate for large models
        
        if free_space < estimated_checkpoint_size * 2:  # Need 2x space for safety
            print(f"⚠️  Low disk space ({free_space / 1e9:.1f}GB free). Cleaning old checkpoints...")
            self.cleanup_old_checkpoints(keep_last_n=2)
        
        # Prepare checkpoint data with CPU tensors to avoid device issues
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                   for k, v in self.optimizer.state_dict().items()},
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save with temporary file to prevent corruption
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        # Save to temporary file first
        torch.save(checkpoint, temp_path)
        
        # If successful, rename to final filename
        temp_path.rename(checkpoint_path)
        
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            best_temp_path = best_path.with_suffix('.tmp')
            
            torch.save(checkpoint, best_temp_path)
            best_temp_path.rename(best_path)
            
            print(f"✓ Saved best checkpoint: {best_path}")
            
        # Clean up old checkpoints to save space
        if epoch > 5:  # Keep some history
            self.cleanup_old_checkpoints(keep_last_n=3)
            
    except Exception as e:
        print(f"❌ Error saving checkpoint: {str(e)}")
        # Try to clean up any temporary files
        try:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            if 'best_temp_path' in locals() and best_temp_path.exists():
                best_temp_path.unlink()
        except:
            pass
        raise e


def cleanup_old_checkpoints(self, keep_last_n: int = 3):
    """Clean up old checkpoints to save disk space."""
    try:
        # Get all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove old checkpoints, keeping the last N
        files_to_remove = checkpoint_files[:-keep_last_n] if len(checkpoint_files) > keep_last_n else []
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                print(f"Cleaned up old checkpoint: {file_path.name}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")
                
    except Exception as e:
        print(f"Error during checkpoint cleanup: {e}")


def generate_sample_fixed(self, batch: Dict[str, Any], num_inference_steps: int = 50) -> torch.Tensor:
    """Generate a sample transition with proper device handling."""
    self.model.eval()
    
    with torch.no_grad():
        # Ensure all inputs are on the correct device
        preceding_spec = batch['preceding_spectrogram'][:1].to(self.device)
        following_spec = batch['following_spectrogram'][:1].to(self.device)
        tempo = batch['avg_tempo'][:1].to(self.device)
        transition_length = batch['transition_length'][:1].to(self.device)
        transition_types = [batch['transition_type'][0]]
        
        # Initialize with random noise on correct device
        target_shape = batch['target_transition_spectrogram'][:1].shape
        generated_transition = torch.randn(target_shape, device=self.device)
        
        # Set scheduler for inference
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Ensure timestep is on correct device
            timestep = t.unsqueeze(0).to(self.device)
            
            # Prepare input
            model_input = torch.cat([preceding_spec, following_spec, generated_transition], dim=1)
            
            # Predict noise
            noise_pred = self.model(
                sample=model_input,
                timestep=timestep,
                tempo=tempo,
                transition_types=transition_types,
                transition_lengths=transition_length,
                return_dict=False
            )
            
            # Update generated transition
            generated_transition = self.scheduler.step(
                noise_pred, t, generated_transition
            ).prev_sample
            
    self.model.train()
    return generated_transition


# Usage: Apply these fixes to your trainer.py by replacing the corresponding methods
print("Fixed trainer methods ready for implementation!")
print("Key fixes:")
print("1. Proper device handling in save_sample_audio")
print("2. Robust checkpoint saving with disk space management")
print("3. Fixed device placement in generate_sample")
print("4. Automatic cleanup of old checkpoints")
