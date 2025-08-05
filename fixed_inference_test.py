#!/usr/bin/env python3
"""
Fixed inference test with proper audio handling
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

def test_fixed_inference():
    """Test inference with fixed audio handling."""
    
    print("Testing DJNet-Diffusion with Fixed Audio Processing")
    print("=" * 60)
    
    # Load configs
    try:
        model_config = load_config('configs/model_config.yaml')
        training_config = load_config('configs/training_config.yaml')
        print("✓ Loaded configurations")
    except Exception as e:
        print(f"✗ Error loading configs: {e}")
        return False
    
    # Check if checkpoint exists
    checkpoint_path = 'checkpoints/best_checkpoint1.pth'
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    print(f"✓ Found checkpoint: {checkpoint_path}")
    
    # Initialize generator
    try:
        generator = DJNetGenerator(
            checkpoint_path=checkpoint_path,
            device=torch.device('cpu')
        )
        print("✓ Initialized DJNet generator")
    except Exception as e:
        print(f"✗ Error initializing generator: {e}")
        return False
    
    # Use the test audio files
    song_a_path = 'source_a.wav'
    song_b_path = 'source_b.wav'
    
    if not (os.path.exists(song_a_path) and os.path.exists(song_b_path)):
        print(f"✗ Test audio files not found: {song_a_path}, {song_b_path}")
        return False
    
    print(f"✓ Using test files: {song_a_path}, {song_b_path}")
    
    # Test different approaches
    test_configs = [
        {"steps": 20, "guidance": 7.5, "length": 2.0, "name": "standard"},
        {"steps": 50, "guidance": 15.0, "length": 2.0, "name": "high_guidance"},
        {"steps": 30, "guidance": 5.0, "length": 3.0, "name": "low_guidance"},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'-'*60}")
        print(f"TEST {i+1}: {config['name']}")
        print(f"Steps: {config['steps']}, Guidance: {config['guidance']}, Length: {config['length']}s")
        print(f"{'-'*60}")
        
        try:
            # Generate just the spectrogram first
            print("1. Generating spectrogram...")
            transition_spec = generator.generate_transition(
                song_a_path=song_a_path,
                song_b_path=song_b_path,
                transition_length=config['length'],
                tempo=120.0,
                transition_type='linear_fade',
                num_inference_steps=config['steps'],
                guidance_scale=config['guidance'],
                seed=42
            )
            
            print(f"   Spectrogram shape: {transition_spec.shape}")
            print(f"   Spectrogram range: [{transition_spec.min():.4f}, {transition_spec.max():.4f}]")
            print(f"   Spectrogram mean: {transition_spec.mean():.4f}")
            
            # Convert to audio using Griffin-Lim with careful handling
            print("2. Converting to audio...")
            
            # Get audio config from the model
            audio_config = generator.config['audio']
            
            # Manual conversion with proper scaling
            if audio_config['normalize_spectrograms']:
                # Denormalize from [-1, 1] to original scale
                spec_min = audio_config['spec_min']
                spec_max = audio_config['spec_max']
                denorm_spec = (transition_spec + 1.0) / 2.0 * (spec_max - spec_min) + spec_min
            else:
                denorm_spec = transition_spec
            
            print(f"   Denormalized range: [{denorm_spec.min():.4f}, {denorm_spec.max():.4f}]")
            
            # Convert from dB to magnitude (if needed)
            if denorm_spec.max() < 10:  # Likely in dB scale
                magnitude_spec = torch.pow(10.0, denorm_spec / 20.0)  # Use 20 instead of 10 for amplitude
            else:
                magnitude_spec = torch.exp(denorm_spec)  # Or use exp if it's log scale
            
            print(f"   Magnitude range: [{magnitude_spec.min():.4f}, {magnitude_spec.max():.4f}]")
            
            # Apply Griffin-Lim
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=audio_config['n_fft'],
                win_length=audio_config['win_length'],
                hop_length=audio_config['hop_length'],
                power=2.0,
                n_iter=64  # More iterations for better quality
            )
            
            # Convert mel to linear spectrogram first
            mel_to_linear = torchaudio.transforms.InverseMelScale(
                n_stft=audio_config['n_fft'] // 2 + 1,
                n_mels=audio_config['n_mels'],
                sample_rate=audio_config['sample_rate']
            )
            
            linear_spec = mel_to_linear(magnitude_spec)
            print(f"   Linear spec range: [{linear_spec.min():.4f}, {linear_spec.max():.4f}]")
            
            # Convert to audio
            audio = griffin_lim(linear_spec)
            
            print(f"   Generated audio shape: {audio.shape}")
            print(f"   Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
            
            # Calculate RMS
            rms = torch.sqrt(torch.mean(audio**2))
            print(f"   Audio RMS: {rms:.6f}")
            
            # Save with different normalizations
            filename_base = f"fixed_test_{i+1}_{config['name']}"
            
            # Version 1: No normalization
            raw_filename = f"{filename_base}_raw.wav"
            torchaudio.save(raw_filename, audio.unsqueeze(0), audio_config['sample_rate'])
            print(f"   ✓ Saved raw: {raw_filename}")
            
            # Version 2: Gentle normalization (preserve dynamic range)
            if rms > 0:
                target_rms = 0.1  # Target RMS
                gentle_audio = audio * (target_rms / rms)
                gentle_audio = torch.clamp(gentle_audio, -0.95, 0.95)  # Soft clipping
                gentle_filename = f"{filename_base}_gentle.wav"
                torchaudio.save(gentle_filename, gentle_audio.unsqueeze(0), audio_config['sample_rate'])
                print(f"   ✓ Saved gentle normalization: {gentle_filename}")
            
            # Version 3: Peak normalization
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                peak_audio = audio * (0.8 / max_val)  # Normalize to 80% of full scale
                peak_filename = f"{filename_base}_peak.wav"
                torchaudio.save(peak_filename, peak_audio.unsqueeze(0), audio_config['sample_rate'])
                print(f"   ✓ Saved peak normalized: {peak_filename}")
            
            print(f"   🎵 Generated audio with RMS: {rms:.6f}")
            
            if rms < 1e-6:
                print("   🔇 STILL SILENT - Check mel-to-linear conversion")
            elif rms < 1e-4:
                print("   🔉 VERY QUIET - Try the gentle normalized version")
            elif rms < 1e-2:
                print("   🔊 QUIET but should be audible")
            else:
                print("   🔊 NORMAL volume")
            
        except Exception as e:
            print(f"✗ Error in test {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare with training data
    print(f"\n{'-'*60}")
    print("TRAINING DATA COMPARISON")
    print(f"{'-'*60}")
    
    try:
        import pandas as pd
        metadata = pd.read_csv('metadata.csv')
        
        if len(metadata) > 0:
            sample_path = metadata.iloc[0]['path']
            target_path = Path(sample_path) / 'target.wav'
            
            if target_path.exists():
                target_audio, _ = torchaudio.load(target_path)
                target_rms = torch.sqrt(torch.mean(target_audio**2))
                target_max = torch.max(torch.abs(target_audio))
                
                print(f"Training target:")
                print(f"  RMS: {target_rms:.6f}")
                print(f"  Max: {target_max:.6f}")
                print(f"  Range: [{target_audio.min():.6f}, {target_audio.max():.6f}]")
            else:
                print("Training target not found for comparison")
        else:
            print("No metadata available")
            
    except Exception as e:
        print(f"Could not load training data: {e}")
    
    print(f"\n{'='*60}")
    print("INFERENCE TEST COMPLETE")
    print("="*60)
    print("Check the generated files:")
    print("- *_raw.wav: Direct model output")
    print("- *_gentle.wav: RMS normalized to 0.1")
    print("- *_peak.wav: Peak normalized to 0.8")
    print("Try listening to all versions to see which works best!")

if __name__ == "__main__":
    test_fixed_inference()
