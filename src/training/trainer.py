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
                # Prepare input
                model_input = torch.cat([preceding_spec, following_spec, generated_transition], dim=1)
                
                # Predict noise
                noise_pred = self.model(
                    sample=model_input,
                    timestep=t.unsqueeze(0),
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
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.lr_scheduler and checkpoint['scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
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
        """Generate and save sample audio for inspection."""
        try:
            # Get a validation batch
            val_batch = next(iter(self.val_loader))
            
            # Generate samples
            for i in range(min(self.config['logging']['num_audio_samples'], len(val_batch['transition_id']))):
                # Create single sample batch
                sample_batch = {k: v[i:i+1] if isinstance(v, torch.Tensor) else [v[i]] 
                               for k, v in val_batch.items()}
                
                # Generate transition
                generated_spec = self.generate_sample(sample_batch)
                
                # Convert to audio (you'll need to implement this)
                # generated_audio = spectrogram_to_audio(generated_spec, self.config)
                
                # Save audio
                # save_path = self.checkpoint_dir / f'samples/epoch_{epoch}_sample_{i}.wav'
                # save_path.parent.mkdir(exist_ok=True)
                # save_audio(generated_audio, save_path, self.config['audio']['sample_rate'])
                
        except Exception as e:
            print(f"Error saving sample audio: {str(e)}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
