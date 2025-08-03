#!/usr/bin/env python3
"""
Manual inference script - specify your own audio files
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference.generator import DJNetGenerator
from src.training.trainer import load_config

def generate_transition(song_a_path, song_b_path, output_path):
    """Generate a transition between two songs."""
    
    print("DJNet-Diffusion Manual Inference")
    print("=" * 40)
    
    # Load configs
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    
    # Initialize generator
    generator = DJNetGenerator(
        checkpoint_path='checkpoints/best_checkpoint.pth',
        device=torch.device('cpu')  # Change to torch.device('cuda') if you have GPU
    )
    
    print(f"Song A: {song_a_path}")
    print(f"Song B: {song_b_path}")
    print(f"Output: {output_path}")
    print("\nGenerating transition...")
    
    # Generate transition
    result_path = generator.generate_transition_audio(
        song_a_path=song_a_path,
        song_b_path=song_b_path,
        output_path=output_path,
        transition_length=3.0,        # 3 second transition
        tempo=128.0,                  # BPM
        transition_type='linear_fade', # or 'bass_swap_eq', 'filter_sweep', etc.
        num_inference_steps=25,       # Higher = better quality, slower
        guidance_scale=7.5,           # Higher = more adherence to conditions
        seed=42                       # For reproducible results
    )
    
    if result_path:
        print(f"✓ Generated: {result_path}")
    else:
        print("✗ Generation failed")

if __name__ == "__main__":
    # Example usage - update these paths
    song_a = "path/to/your/song_a.wav"
    song_b = "path/to/your/song_b.wav"
    output = "my_generated_transition.wav"
    
    # Check if default paths exist, otherwise prompt user
    if not Path(song_a).exists():
        print("Please update the file paths in this script:")
        print(f"  song_a = '{song_a}'")
        print(f"  song_b = '{song_b}'")
        print(f"  output = '{output}'")
        sys.exit(1)
    
    generate_transition(song_a, song_b, output)
