import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.models.diffusion_model import DJNetDiffusionModel
from src.data.dataset import create_dataloaders
from src.utils.audio_utils import spectrogram_to_audio, save_audio


class DJNetTrainer:
    """Trainer class for DJNet diffusion model."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = DJNetDiffusionModel(config).to(device)
        
        # Initialize scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config['scheduler']['num_train_timesteps'],
            beta_start=config['scheduler']['beta_start'],
            beta_end=config['scheduler']['beta_end'],
            beta_schedule=config['scheduler']['beta_schedule'],
            variance_type=config['scheduler']['variance_type'],
            clip_sample=config['scheduler']['clip_sample']
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize scheduler (learning rate)
        if config['optimization']['scheduler'] == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs'],
                eta_min=config['optimization']['min_lr']
            )
        else:
            self.lr_scheduler = None
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging with tensorboard and optionally wandb."""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        self.tb_writer = SummaryWriter(log_dir / 'tensorboard')
        
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"djnet-diffusion-{self.current_epoch}"
            )
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        preceding_spec = batch['preceding_spectrogram'].to(self.device)
        following_spec = batch['following_spectrogram'].to(self.device)
        target_spec = batch['target_transition_spectrogram'].to(self.device)
        tempo = batch['avg_tempo'].to(self.device)
        transition_length = batch['transition_length'].to(self.device)
        transition_types = batch['transition_type']
        
        batch_size = target_spec.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to target spectrograms
        noise = torch.randn_like(target_spec)
        noisy_target = self.scheduler.add_noise(target_spec, noise, timesteps)
        
        # Concatenate context and noisy target as input
        model_input = torch.cat([preceding_spec, following_spec, noisy_target], dim=1)
        
        # Forward pass
        model_output = self.model(
            sample=model_input,
            timestep=timesteps,
            tempo=tempo,
            transition_types=transition_types,
            transition_lengths=transition_length,
            return_dict=False
        )
        
        # Calculate loss (predict noise)
        loss = F.mse_loss(model_output, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config['training']['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_norm']
            )
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self) -> float:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                preceding_spec = batch['preceding_spectrogram'].to(self.device)
                following_spec = batch['following_spectrogram'].to(self.device)
                target_spec = batch['target_transition_spectrogram'].to(self.device)
                tempo = batch['avg_tempo'].to(self.device)
                transition_length = batch['transition_length'].to(self.device)
                transition_types = batch['transition_type']
                
                batch_size = target_spec.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (batch_size,), device=self.device
                ).long()
                
                # Add noise to target spectrograms
                noise = torch.randn_like(target_spec)
                noisy_target = self.scheduler.add_noise(target_spec, noise, timesteps)
                
                # Concatenate context and noisy target as input
                model_input = torch.cat([preceding_spec, following_spec, noisy_target], dim=1)
                
                # Forward pass
                model_output = self.model(
                    sample=model_input,
                    timestep=timesteps,
                    tempo=tempo,
                    transition_types=transition_types,
                    transition_lengths=transition_length,
                    return_dict=False
                )
                
                # Calculate loss
                loss = F.mse_loss(model_output, noise)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def generate_sample(self, batch: Dict[str, Any], num_inference_steps: int = 50) -> torch.Tensor:
        """Generate a sample transition."""
        self.model.eval()
        
        with torch.no_grad():
            # Take first sample from batch
            preceding_spec = batch['preceding_spectrogram'][:1].to(self.device)
            following_spec = batch['following_spectrogram'][:1].to(self.device)
            tempo = batch['avg_tempo'][:1].to(self.device)
            transition_length = batch['transition_length'][:1].to(self.device)
            transition_types = [batch['transition_type'][0]]
            
            # Initialize with random noise
            target_shape = batch['target_transition_spectrogram'][:1].shape
            generated_transition = torch.randn(target_shape, device=self.device)
            
            # Set scheduler for inference
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Denoising loop
            for t in tqdm(self.scheduler.timesteps, desc="Generating"):
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
                
                # Denoise
                generated_transition = self.scheduler.step(
                    noise_pred, t, generated_transition
                ).prev_sample
            
            return generated_transition
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
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
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
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
                best_path = self.checkpoint_dir / 'best_model.pth'
                best_temp_path = best_path.with_suffix('.tmp')
                
                torch.save(checkpoint, best_temp_path)
                best_temp_path.rename(best_path)
                
                print(f"✓ Saved best checkpoint: {best_path}")
            
            # Save latest checkpoint
            latest_path = self.checkpoint_dir / 'latest_model.pth'
            latest_temp_path = latest_path.with_suffix('.tmp')
            
            torch.save(checkpoint, latest_temp_path)
            latest_temp_path.rename(latest_path)
            
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
                if 'latest_temp_path' in locals() and latest_temp_path.exists():
                    latest_temp_path.unlink()
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint with enhanced error handling."""
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded model state")
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Loaded optimizer state")
            
            # Load scheduler state if available
            if self.lr_scheduler and checkpoint.get('scheduler_state_dict'):
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Loaded scheduler state")
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"✅ Successfully loaded checkpoint from epoch {self.current_epoch}")
            print(f"   Best validation loss: {self.best_val_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {str(e)}")
            raise e
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the checkpoint directory."""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number and get the latest
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        return str(latest_checkpoint)
    
    def auto_resume(self) -> bool:
        """Automatically resume from the latest checkpoint if available."""
        latest_checkpoint = self.find_latest_checkpoint()
        
        if latest_checkpoint:
            print(f"🔄 Auto-resuming from: {Path(latest_checkpoint).name}")
            self.load_checkpoint(latest_checkpoint)
            return True
        else:
            print("🆕 No existing checkpoints found. Starting from scratch.")
            return False
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Log to tensorboard
                if self.global_step % self.config['logging']['log_every_n_steps'] == 0:
                    self.tb_writer.add_scalar('Loss/Train_Step', loss, self.global_step)
                    
                    if self.config['logging']['use_wandb']:
                        wandb.log({'train_loss_step': loss, 'step': self.global_step})
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Validation
            if (epoch + 1) % self.config['training']['validate_every_n_epochs'] == 0:
                val_loss = self.validate()
                
                # Log validation loss
                self.tb_writer.add_scalar('Loss/Train_Epoch', avg_epoch_loss, epoch)
                self.tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
                
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'train_loss_epoch': avg_epoch_loss,
                        'val_loss': val_loss,
                        'epoch': epoch
                    })
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Save best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
                
                # Generate and save sample audio
                if self.config['logging']['save_audio_samples']:
                    self.save_sample_audio(epoch)
            
            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch + 1, is_best)
        
        print("Training completed!")
    
    def save_sample_audio(self, epoch: int):
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
