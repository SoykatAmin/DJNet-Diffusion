import torch
import torchaudio
import librosa
import numpy as np
from typing import Optional, Dict, Any
import soundfile as sf


def spectrogram_to_audio(
    spectrogram: torch.Tensor,
    config: Dict[str, Any],
    method: str = 'griffin_lim'
) -> torch.Tensor:
    """Convert mel spectrogram back to audio."""
    
    # Remove batch dimension if present
    if spectrogram.dim() == 4:
        spectrogram = spectrogram.squeeze(0)
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.squeeze(0)
    
    # Denormalize spectrogram
    if config['audio']['normalize_spectrograms']:
        spec_min = config['audio']['spec_min']
        spec_max = config['audio']['spec_max']
        # Convert from [-1, 1] back to [spec_min, spec_max]
        spectrogram = (spectrogram + 1.0) / 2.0 * (spec_max - spec_min) + spec_min
    
    # Convert from dB back to magnitude
    # Use proper dB to amplitude conversion: magnitude = 10^(dB/20)
    # Clamp dB values to prevent overflow
    clamped_db = torch.clamp(spectrogram, min=-100, max=20)
    spectrogram_magnitude = torch.pow(10.0, clamped_db / 20.0)
    
    if method == 'griffin_lim':
        # Use Griffin-Lim algorithm for reconstruction
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config['audio']['n_fft'],
            win_length=config['audio']['win_length'],
            hop_length=config['audio']['hop_length'],
            power=2.0,
            n_iter=32
        )
        
        audio = griffin_lim(spectrogram_magnitude)
        
    elif method == 'mel_to_linear':
        # Convert mel spectrogram to linear spectrogram first, then Griffin-Lim
        mel_to_linear = torchaudio.transforms.InverseMelScale(
            n_stft=config['audio']['n_fft'] // 2 + 1,
            n_mels=config['audio']['n_mels'],
            sample_rate=config['audio']['sample_rate']
        )
        
        linear_spec = mel_to_linear(spectrogram_magnitude)
        
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config['audio']['n_fft'],
            win_length=config['audio']['win_length'],
            hop_length=config['audio']['hop_length'],
            power=2.0,
            n_iter=32
        )
        
        audio = griffin_lim(linear_spec)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
    
    return audio


def mel_to_linear(mel_spec: torch.Tensor, sr: int = 22050, n_fft: int = 1024, n_mels: int = 128) -> torch.Tensor:
    """Convert mel spectrogram to linear spectrogram."""
    mel_to_linear_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sr,
        f_min=0,
        f_max=sr // 2,
        mel_scale="htk"
    )
    return mel_to_linear_transform(mel_spec)


def mel_to_audio(mel_spec: torch.Tensor, sr: int = 22050, n_fft: int = 1024, 
                hop_length: int = 256, n_iter: int = 32) -> torch.Tensor:
    """Convert mel spectrogram directly to audio."""
    
    # Remove extra dimensions
    if mel_spec.dim() > 2:
        mel_spec = mel_spec.squeeze()
    
    # Convert mel to linear
    linear_spec = mel_to_linear(mel_spec, sr=sr, n_fft=n_fft)
    
    # Apply Griffin-Lim
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=n_iter,
        momentum=0.99,
        rand_init=True
    )
    
    audio = griffin_lim(linear_spec)
    return audio


def save_audio(audio: torch.Tensor, filepath: str, sample_rate: int):
    """Save audio tensor to file with proper handling."""
    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio
    
    # Ensure audio is 1D or 2D
    if audio_np.ndim > 2:
        audio_np = audio_np.squeeze()
    
    # Handle 2D case (stereo or batch)
    if audio_np.ndim == 2:
        if audio_np.shape[0] == 1:  # Single channel
            audio_np = audio_np.squeeze(0)
        elif audio_np.shape[1] == 1:  # Single channel, transposed
            audio_np = audio_np.squeeze(1)
    
    # Check for NaN or Inf
    if np.isnan(audio_np).any() or np.isinf(audio_np).any():
        print(f"Warning: Audio contains NaN or Inf values, replacing with zeros")
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Gentle normalization - only if audio is very loud
    max_val = np.max(np.abs(audio_np))
    if max_val > 1.0:
        # Soft compression instead of hard clipping
        audio_np = audio_np / (max_val * 1.1)  # Slight headroom
    
    # Save using soundfile
    sf.write(filepath, audio_np, sample_rate)


def load_audio_segment(
    filepath: str,
    start_time: float,
    duration: float,
    sample_rate: int = 22050
) -> torch.Tensor:
    """Load a specific segment of audio file."""
    
    # Calculate frame parameters
    start_frame = int(start_time * sample_rate)
    num_frames = int(duration * sample_rate)
    
    # Load audio segment
    audio, sr = torchaudio.load(
        filepath,
        frame_offset=start_frame,
        num_frames=num_frames
    )
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    return audio.squeeze(0)


def compute_audio_features(audio: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    """Compute basic audio features."""
    
    # Convert to numpy for librosa
    audio_np = audio.detach().cpu().numpy()
    
    # Compute features
    features = {}
    
    # RMS energy
    features['rms'] = float(np.sqrt(np.mean(audio_np ** 2)))
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio_np, sr=sample_rate
    )
    features['spectral_centroid'] = float(np.mean(spectral_centroids))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_np)
    features['zero_crossing_rate'] = float(np.mean(zcr))
    
    # Tempo estimation
    try:
        tempo, _ = librosa.beat.beat_track(y=audio_np, sr=sample_rate)
        features['tempo'] = float(tempo)
    except:
        features['tempo'] = 120.0  # Default tempo
    
    return features


def apply_fade(audio: torch.Tensor, fade_in_samples: int, fade_out_samples: int) -> torch.Tensor:
    """Apply fade in and fade out to audio."""
    
    audio = audio.clone()
    length = audio.shape[0]
    
    # Apply fade in
    if fade_in_samples > 0:
        fade_in_samples = min(fade_in_samples, length // 2)
        fade_in = torch.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in
    
    # Apply fade out
    if fade_out_samples > 0:
        fade_out_samples = min(fade_out_samples, length // 2)
        fade_out = torch.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out
    
    return audio


def normalize_audio(audio: torch.Tensor, target_level: float = -3.0) -> torch.Tensor:
    """Normalize audio to target dB level."""
    
    # Calculate current RMS level in dB
    rms = torch.sqrt(torch.mean(audio ** 2))
    current_db = 20 * torch.log10(rms + 1e-8)
    
    # Calculate gain needed
    gain_db = target_level - current_db
    gain_linear = torch.pow(10, gain_db / 20)
    
    # Apply gain
    normalized_audio = audio * gain_linear
    
    # Ensure no clipping
    max_val = torch.max(torch.abs(normalized_audio))
    if max_val > 1.0:
        normalized_audio = normalized_audio / max_val
    
    return normalized_audio


def concatenate_audio_with_crossfade(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    crossfade_duration: float,
    sample_rate: int
) -> torch.Tensor:
    """Concatenate two audio segments with crossfade."""
    
    crossfade_samples = int(crossfade_duration * sample_rate)
    crossfade_samples = min(crossfade_samples, min(len(audio1), len(audio2)) // 2)
    
    if crossfade_samples <= 0:
        return torch.cat([audio1, audio2])
    
    # Extract crossfade regions
    audio1_fade = audio1[-crossfade_samples:].clone()
    audio2_fade = audio2[:crossfade_samples].clone()
    
    # Create fade curves
    fade_out = torch.linspace(1, 0, crossfade_samples)
    fade_in = torch.linspace(0, 1, crossfade_samples)
    
    # Apply fades and mix
    mixed_region = audio1_fade * fade_out + audio2_fade * fade_in
    
    # Concatenate
    result = torch.cat([
        audio1[:-crossfade_samples],
        mixed_region,
        audio2[crossfade_samples:]
    ])
    
    return result
