#!/usr/bin/env python3
"""
Simple debug script to analyze the silent output issue
"""

import torch
import torchaudio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def simple_debug():
    """Simple debug analysis."""
    print("DJNet Debug Analysis")
    print("=" * 50)
    
    try:
        from src.inference.generator import DJNetGenerator
        from src.training.trainer import load_config
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return
    
    # Check audio files
    if not os.path.exists('source_a.wav'):
        print("✗ source_a.wav not found")
        return
    if not os.path.exists('source_b.wav'):
        print("✗ source_b.wav not found")
        return
    print("✓ Source audio files found")
    
    # Load and analyze existing generated audio
    generated_files = ['test_generated_transition.wav', 'debug_generated_transition.wav']
    
    for filename in generated_files:
        if os.path.exists(filename):
            print(f"\nAnalyzing {filename}:")
            try:
                waveform, sr = torchaudio.load(filename)
                print(f"  Shape: {waveform.shape}")
                print(f"  Sample rate: {sr}")
                print(f"  Duration: {waveform.shape[1]/sr:.2f} seconds")
                
                # Calculate RMS energy
                rms = torch.sqrt(torch.mean(waveform**2))
                print(f"  RMS energy: {rms:.6f}")
                
                # Calculate max amplitude
                max_amp = torch.max(torch.abs(waveform))
                print(f"  Max amplitude: {max_amp:.6f}")
                
                # Check if it's effectively silent
                if rms < 1e-6:
                    print(f"  🔇 SILENT (RMS < 1e-6)")
                elif rms < 1e-4:
                    print(f"  🔉 VERY QUIET (RMS < 1e-4)")
                elif rms < 1e-2:
                    print(f"  🔊 QUIET but audible (RMS < 1e-2)")
                else:
                    print(f"  🔊 NORMAL volume")
                
                # Try amplifying and saving
                if rms > 0:
                    amplified = waveform * (0.1 / rms)  # Normalize to 0.1 RMS
                    amplified = torch.clamp(amplified, -1.0, 1.0)
                    amp_filename = f"amplified_{filename}"
                    torchaudio.save(amp_filename, amplified, sr)
                    print(f"  ✓ Saved amplified version: {amp_filename}")
                
            except Exception as e:
                print(f"  ✗ Error analyzing {filename}: {e}")
    
    # Compare with source audio
    print(f"\nSource audio analysis:")
    for source in ['source_a.wav', 'source_b.wav']:
        try:
            waveform, sr = torchaudio.load(source)
            rms = torch.sqrt(torch.mean(waveform**2))
            print(f"  {source}: RMS={rms:.6f}, Max={torch.max(torch.abs(waveform)):.6f}")
        except Exception as e:
            print(f"  Error with {source}: {e}")

if __name__ == "__main__":
    simple_debug()
