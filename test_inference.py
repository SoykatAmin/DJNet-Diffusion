#!/usr/bin/env python3
"""
Simple inference test script for DJNet-Diffusion
"""

import torch
import torchaudio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.inference.generator import DJNetGenerator
from src.training.trainer import load_config

def test_inference():
    """Test inference with the trained model."""
    
    print("Testing DJNet-Diffusion Inference")
    print("=" * 50)
    
    # Load configs
    try:
        model_config = load_config('configs/model_config.yaml')
        training_config = load_config('configs/training_config.yaml')
        print("✓ Loaded configurations")
    except Exception as e:
        print(f"✗ Error loading configs: {e}")
        return False
    
    # Check if checkpoint exists
    checkpoint_path = 'checkpoints/best_checkpoint2.pth'
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    print(f"✓ Found checkpoint: {checkpoint_path}")
    
    # Initialize generator
    try:
        generator = DJNetGenerator(
            checkpoint_path=checkpoint_path,
            device=torch.device('cpu')  # Use CPU for local testing
        )
        print("✓ Initialized DJNet generator")
    except Exception as e:
        print(f"✗ Error initializing generator: {e}")
        return False
    
    # Find some test audio files from your dataset
    # Check multiple possible locations
    possible_data_roots = [
        Path('./output/djnet_dataset_20k'),
        Path('./dataset'),
        Path('../DJNet-Dataset/output/djnet_dataset_20k'),
        Path('./data')
    ]
    
    data_root = None
    for path in possible_data_roots:
        if path.exists():
            data_root = path
            break
    
    if data_root is None:
        print("✗ Dataset directory not found in common locations:")
        for path in possible_data_roots:
            print(f"  - {path}")
        print("Please update the data_root path in this script to point to your dataset")
        
        # Try to use the first transition from metadata as fallback
        try:
            import pandas as pd
            metadata = pd.read_csv('metadata.csv')
            first_path = metadata.iloc[0]['path']
            transition_dir = Path(first_path)
            if transition_dir.exists():
                data_root = transition_dir.parent
                print(f"✓ Found dataset using metadata: {data_root}")
            else:
                print(f"✗ Path from metadata doesn't exist: {transition_dir}")
                return False
        except Exception as e:
            print(f"✗ Could not read metadata.csv: {e}")
            return False
    
    # Find a sample transition directory
    transition_dirs = list(data_root.glob('transition_*'))
    if not transition_dirs:
        print("✗ No transition directories found in dataset")
        return False
    
    # Use the first available transition
    test_transition = transition_dirs[0]
    song_a_path = test_transition / 'source_a.wav'
    song_b_path = test_transition / 'source_b.wav'
    
    if not (song_a_path.exists() and song_b_path.exists()):
        print(f"✗ Test audio files not found in {test_transition}")
        return False
    
    print(f"✓ Using test files from: {test_transition.name}")
    print(f"  - Song A: {song_a_path.name}")
    print(f"  - Song B: {song_b_path.name}")
    
    # Test generation
    output_path = 'test_generated_transition.wav'
    
    try:
        print("\nGenerating transition...")
        print("This may take a few minutes on CPU...")
        
        result_path = generator.generate_transition_audio(
            song_a_path=str(song_a_path),
            song_b_path=str(song_b_path),
            output_path=output_path,
            transition_length=4.0,  # 2 second transition
            tempo=120.0,
            transition_type='linear_fade',  # Use a simple type for testing
            # transition_type='linear_fade',  # or 'bass_swap_eq', 'filter_sweep', etc.
            num_inference_steps=20,  # Reduced for faster CPU inference
            guidance_scale=7.5,
            seed=42,
            add_context=False,  # Disable context to avoid crossfade issues
            crossfade_duration=0.0  # Disable crossfade
        )
        
        if result_path and os.path.exists(result_path):
            print(f"✓ Successfully generated transition: {result_path}")
            
            # Get audio info
            info = torchaudio.info(result_path)
            duration = info.num_frames / info.sample_rate
            print(f"  - Duration: {duration:.2f} seconds")
            print(f"  - Sample rate: {info.sample_rate} Hz")
            print(f"  - Channels: {info.num_channels}")
            
            return True
        else:
            print("✗ Generation failed - no output file created")
            return False
            
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_inference()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 Inference test completed successfully!")
        print("Check the generated file: test_generated_transition.wav")
    else:
        print("\n" + "=" * 50)
        print("❌ Inference test failed")
        print("Please check the error messages above")
