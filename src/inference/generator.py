import torch
from diffusers import DDPMScheduler
from pathlib import Path
import torchaudio
from typing import Dict, Any, Optional, List
import numpy as np

from src.models.diffusion_model import DJNetDiffusionModel
from src.data.dataset import SpectrogramProcessor
from src.utils.audio_utils import spectrogram_to_audio, save_audio, apply_fade


class DJNetGenerator:
    """Generator class for creating transitions using trained DJNet model."""
    
    def __init__(self, checkpoint_path: str, device: torch.device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = DJNetDiffusionModel(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize scheduler for inference
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config['scheduler']['num_train_timesteps'],
            beta_start=self.config['scheduler']['beta_start'],
            beta_end=self.config['scheduler']['beta_end'],
            beta_schedule=self.config['scheduler']['beta_schedule'],
            variance_type=self.config['scheduler']['variance_type'],
            clip_sample=self.config['scheduler']['clip_sample']
        )
        
        # Initialize spectrogram processor
        self.spec_processor = SpectrogramProcessor(self.config)
        
        print(f"Loaded DJNet model from {checkpoint_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs")
    
    def prepare_context(
        self,
        song_a_path: str,
        song_b_path: str,
        song_a_end_duration: Optional[float] = None,
        song_b_start_duration: Optional[float] = None
    ) -> tuple:
        """Prepare context spectrograms from input songs."""
        
        context_duration = self.config['audio']['context_duration']
        
        if song_a_end_duration is None:
            song_a_end_duration = context_duration
        if song_b_start_duration is None:
            song_b_start_duration = context_duration
        
        # Load audio files
        song_a_audio, sr_a = torchaudio.load(song_a_path)
        song_b_audio, sr_b = torchaudio.load(song_b_path)
        
        # Convert to mono if needed
        if song_a_audio.shape[0] > 1:
            song_a_audio = torch.mean(song_a_audio, dim=0, keepdim=True)
        if song_b_audio.shape[0] > 1:
            song_b_audio = torch.mean(song_b_audio, dim=0, keepdim=True)
        
        # Resample if needed
        target_sr = self.config['audio']['sample_rate']
        if sr_a != target_sr:
            resampler = torchaudio.transforms.Resample(sr_a, target_sr)
            song_a_audio = resampler(song_a_audio)
        if sr_b != target_sr:
            resampler = torchaudio.transforms.Resample(sr_b, target_sr)
            song_b_audio = resampler(song_b_audio)
        
        # Extract end of song A
        song_a_samples = int(song_a_end_duration * target_sr)
        if song_a_audio.shape[1] >= song_a_samples:
            preceding_audio = song_a_audio[:, -song_a_samples:]
        else:
            # Pad if song is too short
            padding = song_a_samples - song_a_audio.shape[1]
            preceding_audio = torch.nn.functional.pad(song_a_audio, (padding, 0))
        
        # Extract start of song B
        song_b_samples = int(song_b_start_duration * target_sr)
        if song_b_audio.shape[1] >= song_b_samples:
            following_audio = song_b_audio[:, :song_b_samples]
        else:
            # Pad if song is too short
            padding = song_b_samples - song_b_audio.shape[1]
            following_audio = torch.nn.functional.pad(song_b_audio, (0, padding))
        
        # Convert to spectrograms
        preceding_spec = self.spec_processor.audio_to_spectrogram(preceding_audio.squeeze(0))
        following_spec = self.spec_processor.audio_to_spectrogram(following_audio.squeeze(0))
        
        # Add batch and channel dimensions
        preceding_spec = preceding_spec.unsqueeze(0).unsqueeze(0)  # (1, 1, mels, time)
        following_spec = following_spec.unsqueeze(0).unsqueeze(0)  # (1, 1, mels, time)
        
        return preceding_spec, following_spec
    
    def generate_transition(
        self,
        song_a_path: str,
        song_b_path: str,
        transition_length: float = 8.0,
        tempo: float = 120.0,
        transition_type: str = 'linear_fade',
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate a transition between two songs."""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Prepare context
        preceding_spec, following_spec = self.prepare_context(song_a_path, song_b_path)
        preceding_spec = preceding_spec.to(self.device)
        following_spec = following_spec.to(self.device)
        
        # Prepare conditioning
        tempo_tensor = torch.tensor([tempo], dtype=torch.float32, device=self.device)
        length_tensor = torch.tensor([transition_length], dtype=torch.float32, device=self.device)
        transition_types = [transition_type]
        
        # Initialize with random noise
        max_transition_duration = self.config['audio']['max_transition_duration']
        transition_frames = int(max_transition_duration * self.config['audio']['sample_rate'] / 
                               self.config['audio']['hop_length']) + 1
        
        noise_shape = (1, 1, self.config['audio']['n_mels'], transition_frames)
        generated_transition = torch.randn(noise_shape, device=self.device)
        
        # Set up scheduler for inference
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Prepare model input
            model_input = torch.cat([preceding_spec, following_spec, generated_transition], dim=1)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(
                    sample=model_input,
                    timestep=t.unsqueeze(0),
                    tempo=tempo_tensor,
                    transition_types=transition_types,
                    transition_lengths=length_tensor,
                    return_dict=False
                )
            
            # Apply guidance (if > 1.0)
            if guidance_scale > 1.0:
                # Unconditional prediction (with empty conditioning)
                model_input_uncond = torch.cat([
                    torch.zeros_like(preceding_spec), 
                    torch.zeros_like(following_spec), 
                    generated_transition
                ], dim=1)
                
                with torch.no_grad():
                    noise_pred_uncond = self.model(
                        sample=model_input_uncond,
                        timestep=t.unsqueeze(0),
                        tempo=torch.zeros_like(tempo_tensor),
                        transition_types=['linear_fade'],  # Default type
                        transition_lengths=torch.ones_like(length_tensor) * 4.0,  # Default length
                        return_dict=False
                    )
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # Denoise
            generated_transition = self.scheduler.step(
                noise_pred, t, generated_transition
            ).prev_sample
        
        return generated_transition
    
    def generate_transition_audio(
        self,
        song_a_path: str,
        song_b_path: str,
        output_path: str,
        transition_length: float = 8.0,
        tempo: float = 120.0,
        transition_type: str = 'linear_fade',
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        add_context: bool = True,
        crossfade_duration: float = 0.5
    ) -> str:
        """Generate transition and save as audio file."""
        
        # Generate transition spectrogram
        transition_spec = self.generate_transition(
            song_a_path=song_a_path,
            song_b_path=song_b_path,
            transition_length=transition_length,
            tempo=tempo,
            transition_type=transition_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # Convert to audio
        transition_audio = spectrogram_to_audio(transition_spec, self.config)
        
        # Trim to desired length
        target_samples = int(transition_length * self.config['audio']['sample_rate'])
        if transition_audio.shape[0] > target_samples:
            transition_audio = transition_audio[:target_samples]
        elif transition_audio.shape[0] < target_samples:
            padding = target_samples - transition_audio.shape[0]
            transition_audio = torch.nn.functional.pad(transition_audio, (0, padding))
        
        if add_context:
            # Load context audio
            context_duration = self.config['audio']['context_duration']
            
            # Load end of song A
            song_a_audio, sr_a = torchaudio.load(song_a_path)
            if song_a_audio.shape[0] > 1:
                song_a_audio = torch.mean(song_a_audio, dim=0)
            if sr_a != self.config['audio']['sample_rate']:
                resampler = torchaudio.transforms.Resample(sr_a, self.config['audio']['sample_rate'])
                song_a_audio = resampler(song_a_audio)
            
            context_samples = int(context_duration * self.config['audio']['sample_rate'])
            if song_a_audio.shape[0] >= context_samples:
                song_a_end = song_a_audio[-context_samples:]
            else:
                padding = context_samples - song_a_audio.shape[0]
                song_a_end = torch.nn.functional.pad(song_a_audio, (padding, 0))
            
            # Load start of song B
            song_b_audio, sr_b = torchaudio.load(song_b_path)
            if song_b_audio.shape[0] > 1:
                song_b_audio = torch.mean(song_b_audio, dim=0)
            if sr_b != self.config['audio']['sample_rate']:
                resampler = torchaudio.transforms.Resample(sr_b, self.config['audio']['sample_rate'])
                song_b_audio = resampler(song_b_audio)
            
            if song_b_audio.shape[0] >= context_samples:
                song_b_start = song_b_audio[:context_samples]
            else:
                padding = context_samples - song_b_audio.shape[0]
                song_b_start = torch.nn.functional.pad(song_b_audio, (0, padding))
            
            # Apply crossfades
            crossfade_samples = int(crossfade_duration * self.config['audio']['sample_rate'])
            
            # Crossfade song A end with transition start
            if crossfade_samples > 0:
                fade_out = torch.linspace(1, 0, crossfade_samples)
                fade_in = torch.linspace(0, 1, crossfade_samples)
                
                song_a_end[-crossfade_samples:] *= fade_out
                transition_audio[:crossfade_samples] *= fade_in
                transition_start_mixed = song_a_end[-crossfade_samples:] + transition_audio[:crossfade_samples]
                
                # Crossfade transition end with song B start
                transition_audio[-crossfade_samples:] *= fade_out
                song_b_start[:crossfade_samples] *= fade_in
                transition_end_mixed = transition_audio[-crossfade_samples:] + song_b_start[:crossfade_samples]
                
                # Combine all parts
                final_audio = torch.cat([
                    song_a_end[:-crossfade_samples],
                    transition_start_mixed,
                    transition_audio[crossfade_samples:-crossfade_samples],
                    transition_end_mixed,
                    song_b_start[crossfade_samples:]
                ])
            else:
                final_audio = torch.cat([song_a_end, transition_audio, song_b_start])
        else:
            final_audio = transition_audio
        
        # Normalize audio
        final_audio = final_audio / max(abs(final_audio.max()), abs(final_audio.min()))
        
        # Save audio
        save_audio(final_audio, output_path, self.config['audio']['sample_rate'])
        
        print(f"Generated transition saved to {output_path}")
        return output_path
    
    def batch_generate(
        self,
        song_pairs: List[tuple],
        output_dir: str,
        **generation_kwargs
    ) -> List[str]:
        """Generate transitions for multiple song pairs."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_paths = []
        
        for i, (song_a, song_b) in enumerate(song_pairs):
            output_path = output_dir / f"transition_{i:04d}.wav"
            
            try:
                generated_path = self.generate_transition_audio(
                    song_a_path=song_a,
                    song_b_path=song_b,
                    output_path=str(output_path),
                    **generation_kwargs
                )
                output_paths.append(generated_path)
                print(f"Generated transition {i+1}/{len(song_pairs)}")
                
            except Exception as e:
                print(f"Error generating transition {i}: {str(e)}")
                output_paths.append(None)
        
        return output_paths
