#!/usr/bin/env python3
"""
Debug inference script to analyze what the model is generating
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference.generator import DJNetGenerator
from src.training.trainer import load_config

def debug_inference():
    """Debug what the model is generating."""
    
    print("DJNet-Diffusion Debug Analysis")
    print("=" * 50)
    
    # Load configs
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    
    # Initialize generator
    checkpoint_path = 'checkpoints/best_checkpoint.pth'
    generator = DJNetGenerator(
        checkpoint_path=checkpoint_path,
        device=torch.device('cpu')
    )
    
    # Use the test audio files
    song_a_path = 'source_a.wav'
    song_b_path = 'source_b.wav'
    
    print("\n1. Generating transition spectrogram...")
    
    # Generate just the spectrogram (without audio conversion)
    transition_spec = generator.generate_transition(
        song_a_path=song_a_path,
        song_b_path=song_b_path,
        transition_length=2.0,
        tempo=120.0,
        transition_type='linear_fade',
        num_inference_steps=20,
        guidance_scale=1.0
    )
    
    print(f"Generated spectrogram shape: {transition_spec.shape}")
    print(f"Spectrogram range: [{transition_spec.min():.4f}, {transition_spec.max():.4f}]")
    print(f"Spectrogram mean: {transition_spec.mean():.4f}")
    print(f"Spectrogram std: {transition_spec.std():.4f}")
    
    # Check if spectrogram is all zeros or very close to zero
    if torch.abs(transition_spec).max() < 1e-6:
        print("⚠️  WARNING: Generated spectrogram is essentially all zeros!")
        print("   This indicates the model hasn't learned to generate meaningful audio.")
        print("   This is expected with only 20 training samples.")
    elif torch.abs(transition_spec).max() < 0.01:
        print("⚠️  WARNING: Generated spectrogram has very low amplitude.")
        print("   The model might be generating very quiet audio.")
    else:
        print("✓ Spectrogram has reasonable amplitude range.")
    
    # Analyze spectrogram content
    spec_np = transition_spec.squeeze().numpy()
    
    print(f"\n2. Spectrogram Analysis:")
    print(f"   - Non-zero elements: {np.count_nonzero(spec_np)} / {spec_np.size}")
    print(f"   - Zero ratio: {np.count_nonzero(spec_np == 0) / spec_np.size:.2%}")
    
    # Try to generate audio
    print(f"\n3. Converting to audio...")
    try:
        # Import audio utils functions directly
        from src.utils.audio_utils import spectrogram_to_audio
        
        # Convert to audio using mel_to_linear method
        audio = spectrogram_to_audio(transition_spec.squeeze(), model_config, method='mel_to_linear')
        print(f"Generated audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
        print(f"Audio RMS: {torch.sqrt(torch.mean(audio**2)):.6f}")
        
        if torch.abs(audio).max() < 1e-6:
            print("⚠️  WARNING: Generated audio is essentially silent!")
        else:
            print("✓ Audio has some amplitude.")
            
        # Save the audio
        output_path = 'debug_generated_transition.wav'
        torchaudio.save(output_path, audio.unsqueeze(0), model_config['audio']['sample_rate'])
        print(f"✓ Saved debug audio to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error converting to audio: {e}")
        import traceback
        traceback.print_exc()
    
    # Save spectrogram plot
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(spec_np, aspect='auto', origin='lower')
        plt.title('Generated Spectrogram')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.plot(np.mean(spec_np, axis=0))
        plt.title('Mean Frequency Content Over Time')
        plt.xlabel('Time Frame')
        plt.ylabel('Mean Amplitude')
        
        plt.tight_layout()
        plt.savefig('debug_spectrogram.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved spectrogram plot to: debug_spectrogram.png")
        plt.close()
        
    except Exception as e:
        print(f"Note: Could not save plot (matplotlib not available): {e}")
    
    print(f"\n4. Training Data Analysis:")
    print(f"   The model was trained on only 20 transitions, which is extremely limited.")
    print(f"   Typical requirements:")
    print(f"   - Minimum for basic results: ~1,000 samples")
    print(f"   - Good results: ~10,000+ samples") 
    print(f"   - Your dataset: 20 samples")
    print(f"   ")
    print(f"   Recommendations:")
    print(f"   1. Use your full 20K dataset for proper training")
    print(f"   2. Train for more epochs (100-500)")
    print(f"   3. Monitor training loss to ensure convergence")
    
if __name__ == "__main__":
    debug_inference()
