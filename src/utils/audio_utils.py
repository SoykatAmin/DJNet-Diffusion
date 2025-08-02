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
    spectrogram_magnitude = torch.pow(10.0, spectrogram / 10.0)
    
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


def save_audio(audio: torch.Tensor, filepath: str, sample_rate: int):
    """Save audio tensor to file."""
    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio
    
    # Ensure audio is 1D
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    
    # Normalize to [-1, 1] if needed
    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
    
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
