#!/usr/bin/env python3
"""
Properly fixed inference with correct audio conversion
"""

import torch
import torchaudio
import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference.generator import DJNetGenerator
from src.training.trainer import load_config

def proper_mel_to_audio(mel_spec, config):
    """Properly convert mel spectrogram to audio."""
    
    # Remove extra dimensions
    if mel_spec.dim() == 4:  # [batch, channel, freq, time]
        mel_spec = mel_spec.squeeze(0).squeeze(0)
    elif mel_spec.dim() == 3:  # [channel, freq, time] or [batch, freq, time]
        mel_spec = mel_spec.squeeze(0)
    
    print(f"Working with mel_spec shape: {mel_spec.shape}")
    print(f"Mel spec range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    
    # Handle normalization carefully
    if config['audio']['normalize_spectrograms']:
        # The spectrograms are normalized to [-1, 1]
        # Convert back to original dB range
        spec_min = config['audio']['spec_min']  # Should be around -80
        spec_max = config['audio']['spec_max']  # Should be around 0
        
        # Denormalize from [-1, 1] to [spec_min, spec_max]
        denorm_spec = (mel_spec + 1.0) / 2.0 * (spec_max - spec_min) + spec_min
    else:
        denorm_spec = mel_spec
    
    print(f"Denormalized range: [{denorm_spec.min():.4f}, {denorm_spec.max():.4f}]")
    
    # Convert from dB to linear scale (magnitude)
    # Use proper dB conversion: magnitude = 10^(dB/20) for amplitude
    # Clamp the dB values to prevent overflow
    clamped_db = torch.clamp(denorm_spec, min=-100, max=20)  # Reasonable dB range
    magnitude_spec = torch.pow(10.0, clamped_db / 20.0)
    
    print(f"Magnitude range: [{magnitude_spec.min():.4f}, {magnitude_spec.max():.4f}]")
    
    # Convert mel to linear spectrogram
    mel_to_linear = torchaudio.transforms.InverseMelScale(
        n_stft=config['audio']['n_fft'] // 2 + 1,
        n_mels=config['audio']['n_mels'],
        sample_rate=config['audio']['sample_rate'],
        f_min=config['audio'].get('f_min', 0),
        f_max=config['audio'].get('f_max', config['audio']['sample_rate'] // 2),
        mel_scale="htk"
    )
    
    linear_spec = mel_to_linear(magnitude_spec)
    print(f"Linear spec range: [{linear_spec.min():.4f}, {linear_spec.max():.4f}]")
    
    # Apply Griffin-Lim
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=config['audio']['n_fft'],
        win_length=config['audio'].get('win_length', config['audio']['n_fft']),
        hop_length=config['audio']['hop_length'],
        power=2.0,  # Use power=2.0 for power spectrograms
        n_iter=64,  # More iterations for better quality
        momentum=0.99,
        length=None,
        rand_init=True
    )
    
    # Convert to audio
    audio = griffin_lim(linear_spec)
    
    print(f"Generated audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    
    return audio

def test_proper_inference():
    """Test inference with properly fixed audio conversion."""
    
    print("Testing DJNet with PROPER Audio Conversion")
    print("=" * 60)
    
    # Load configs
    model_config = load_config('configs/model_config.yaml')
    print("✓ Loaded configurations")
    
    # Initialize generator
    checkpoint_path = 'checkpoints/best_checkpoint1.pth'
    generator = DJNetGenerator(checkpoint_path=checkpoint_path, device=torch.device('cpu'))
    print("✓ Initialized generator")
    
    # Generate a spectrogram
    print("\nGenerating transition spectrogram...")
    transition_spec = generator.generate_transition(
        song_a_path='source_a.wav',
        song_b_path='source_b.wav',
        transition_length=2.0,
        tempo=120.0,
        transition_type='linear_fade',
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=42
    )
    
    print(f"\nGenerated spectrogram: {transition_spec.shape}")
    print(f"Spec range: [{transition_spec.min():.4f}, {transition_spec.max():.4f}]")
    
    # Convert to audio using proper method
    print("\nConverting to audio...")
    audio = proper_mel_to_audio(transition_spec, generator.config)
    
    # Calculate audio statistics
    rms = torch.sqrt(torch.mean(audio**2))
    max_val = torch.max(torch.abs(audio))
    
    print(f"\nAudio statistics:")
    print(f"- Shape: {audio.shape}")
    print(f"- RMS: {rms:.6f}")
    print(f"- Max: {max_val:.6f}")
    print(f"- Range: [{audio.min():.6f}, {audio.max():.6f}]")
    
    # Save different versions
    sample_rate = generator.config['audio']['sample_rate']
    
    # Version 1: Raw audio
    if not torch.isnan(audio).any() and not torch.isinf(audio).any():
        torchaudio.save('proper_raw_audio.wav', audio.unsqueeze(0), sample_rate)
        print("✓ Saved: proper_raw_audio.wav")
        
        # Version 2: Normalized to reasonable level
        if rms > 0:
            target_rms = 0.05  # Moderate level
            normalized_audio = audio * (target_rms / rms)
            normalized_audio = torch.clamp(normalized_audio, -0.95, 0.95)
            torchaudio.save('proper_normalized_audio.wav', normalized_audio.unsqueeze(0), sample_rate)
            print("✓ Saved: proper_normalized_audio.wav")
            
            # Version 3: Peak normalized
            peak_audio = audio * (0.7 / max_val) if max_val > 0 else audio
            torchaudio.save('proper_peak_audio.wav', peak_audio.unsqueeze(0), sample_rate)
            print("✓ Saved: proper_peak_audio.wav")
        
        # Diagnosis
        if rms < 1e-6:
            print("\n🔇 STILL SILENT")
        elif rms < 1e-4:
            print("\n🔉 VERY QUIET - Try the normalized version")
        elif rms < 1e-2:
            print("\n🔊 QUIET but audible")
        else:
            print("\n🔊 NORMAL volume")
    else:
        print("\n❌ Audio contains NaN or Inf values - conversion failed")
        
        # Debug the spectrogram
        print("Debugging spectrogram...")
        print(f"Has NaN in spec: {torch.isnan(transition_spec).any()}")
        print(f"Has Inf in spec: {torch.isinf(transition_spec).any()}")
        
        # Try a different approach - use the working audio_utils
        print("\nTrying with existing audio_utils...")
        from src.utils.audio_utils import spectrogram_to_audio
        
        try:
            audio_utils_result = spectrogram_to_audio(transition_spec, generator.config)
            print(f"Audio utils result shape: {audio_utils_result.shape}")
            print(f"Audio utils RMS: {torch.sqrt(torch.mean(audio_utils_result**2)):.6f}")
            
            if not torch.isnan(audio_utils_result).any():
                torchaudio.save('audio_utils_result.wav', audio_utils_result.unsqueeze(0), sample_rate)
                print("✓ Saved: audio_utils_result.wav")
            else:
                print("Audio utils also produced NaN")
                
        except Exception as e:
            print(f"Audio utils failed: {e}")
    
    # Compare with source audio
    print(f"\n{'-'*60}")
    print("SOURCE AUDIO COMPARISON")
    print(f"{'-'*60}")
    
    for source_file in ['source_a.wav', 'source_b.wav']:
        source_audio, source_sr = torchaudio.load(source_file)
        source_rms = torch.sqrt(torch.mean(source_audio**2))
        print(f"{source_file}: RMS={source_rms:.6f}")
    
    print(f"\n{'='*60}")
    print("PROPER INFERENCE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_proper_inference()
