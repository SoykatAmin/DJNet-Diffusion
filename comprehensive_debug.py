#!/usr/bin/env python3
"""
Comprehensive debug script to analyze silent output issue
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
from src.utils.audio_utils import mel_to_linear, mel_to_audio

def analyze_model_output():
    """Comprehensive analysis of model output to debug silent audio."""
    
    print("DJNet-Diffusion Comprehensive Debug Analysis")
    print("=" * 60)
    
    # Load configs
    try:
        model_config = load_config('configs/model_config.yaml')
        training_config = load_config('configs/training_config.yaml')
        print("✓ Loaded configurations")
    except Exception as e:
        print(f"✗ Error loading configs: {e}")
        return
    
    # Initialize generator with the 600-transition trained model
    checkpoint_path = 'checkpoints/best_checkpoint1.pth'
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
        
    try:
        generator = DJNetGenerator(
            checkpoint_path=checkpoint_path,
            device=torch.device('cpu')
        )
        print(f"✓ Loaded model from: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Use the test audio files
    song_a_path = 'source_a.wav'
    song_b_path = 'source_b.wav'
    
    if not (os.path.exists(song_a_path) and os.path.exists(song_b_path)):
        print(f"✗ Test audio files not found: {song_a_path}, {song_b_path}")
        return
    
    print(f"✓ Using test files: {song_a_path}, {song_b_path}")
    
    # Step 1: Analyze input audio
    print("\n" + "="*60)
    print("STEP 1: INPUT AUDIO ANALYSIS")
    print("="*60)
    
    waveform_a, sr_a = torchaudio.load(song_a_path)
    waveform_b, sr_b = torchaudio.load(song_b_path)
    
    print(f"Source A: shape={waveform_a.shape}, sample_rate={sr_a}, RMS={torch.sqrt(torch.mean(waveform_a**2)):.4f}")
    print(f"Source B: shape={waveform_b.shape}, sample_rate={sr_b}, RMS={torch.sqrt(torch.mean(waveform_b**2)):.4f}")
    
    # Step 2: Generate raw spectrogram from model
    print("\n" + "="*60)
    print("STEP 2: RAW MODEL OUTPUT ANALYSIS")
    print("="*60)
    
    try:
        print("Generating raw spectrogram from model...")
        
        # Generate just the spectrogram (without audio conversion)
        transition_spec = generator.generate_transition(
            song_a_path=song_a_path,
            song_b_path=song_b_path,
            transition_length=2.0,
            tempo=120.0,
            transition_type='linear_fade',
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42
        )
        
        print(f"Generated spectrogram shape: {transition_spec.shape}")
        print(f"Spectrogram range: [{transition_spec.min():.4f}, {transition_spec.max():.4f}]")
        print(f"Spectrogram mean: {transition_spec.mean():.4f}")
        print(f"Spectrogram std: {transition_spec.std():.4f}")
        
        # Check for NaN or inf values
        if torch.isnan(transition_spec).any():
            print("⚠️  WARNING: Spectrogram contains NaN values!")
        if torch.isinf(transition_spec).any():
            print("⚠️  WARNING: Spectrogram contains infinite values!")
            
        # Step 3: Convert to linear spectrogram
        print("\n" + "="*60)
        print("STEP 3: MEL-TO-LINEAR CONVERSION")
        print("="*60)
        
        # Convert mel to linear
        linear_spec = mel_to_linear(transition_spec, sr=22050, n_fft=1024)
        
        print(f"Linear spectrogram shape: {linear_spec.shape}")
        print(f"Linear spectrogram range: [{linear_spec.min():.4f}, {linear_spec.max():.4f}]")
        print(f"Linear spectrogram mean: {linear_spec.mean():.4f}")
        print(f"Linear spectrogram std: {linear_spec.std():.4f}")
        
        # Step 4: Convert to audio using Griffin-Lim
        print("\n" + "="*60)
        print("STEP 4: SPECTROGRAM-TO-AUDIO CONVERSION")
        print("="*60)
        
        # Convert to audio
        audio_output = mel_to_audio(transition_spec, sr=22050, n_fft=1024, hop_length=256, n_iter=32)
        
        print(f"Generated audio shape: {audio_output.shape}")
        print(f"Generated audio range: [{audio_output.min():.4f}, {audio_output.max():.4f}]")
        print(f"Generated audio RMS: {torch.sqrt(torch.mean(audio_output**2)):.4f}")
        
        # Check for silent output
        audio_energy = torch.sqrt(torch.mean(audio_output**2))
        if audio_energy < 1e-6:
            print("🔇 SILENT OUTPUT DETECTED!")
            print("   Audio energy is extremely low")
        elif audio_energy < 1e-3:
            print("🔉 LOW VOLUME OUTPUT")
            print("   Audio energy is low but not silent")
        else:
            print("🔊 NORMAL VOLUME OUTPUT")
            print("   Audio energy looks good")
        
        # Save debug outputs
        print("\n" + "="*60)
        print("STEP 5: SAVING DEBUG OUTPUTS")
        print("="*60)
        
        # Save spectrogram visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.imshow(transition_spec.numpy(), aspect='auto', origin='lower')
        plt.title('Generated Mel Spectrogram')
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(linear_spec.numpy(), aspect='auto', origin='lower')
        plt.title('Linear Spectrogram')
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.plot(audio_output.numpy().flatten())
        plt.title('Generated Audio Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 2, 4)
        # Histogram of audio values
        plt.hist(audio_output.numpy().flatten(), bins=50, alpha=0.7)
        plt.title('Audio Amplitude Distribution')
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('debug_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save the audio with amplification
        debug_audio_path = 'debug_comprehensive_transition.wav'
        
        # Try amplifying the audio
        amplified_audio = audio_output * 10.0  # Amplify by 10x
        amplified_audio = torch.clamp(amplified_audio, -1.0, 1.0)  # Clamp to prevent clipping
        
        torchaudio.save(debug_audio_path, amplified_audio.unsqueeze(0), 22050)
        
        print(f"✓ Saved amplified audio: {debug_audio_path}")
        print(f"✓ Saved analysis plots: debug_comprehensive_analysis.png")
        
        # Test different generation parameters
        print("\n" + "="*60)
        print("STEP 6: TESTING DIFFERENT PARAMETERS")
        print("="*60)
        
        test_params = [
            {"steps": 50, "guidance": 7.5, "type": "linear_fade"},
            {"steps": 20, "guidance": 15.0, "type": "bass_swap_eq"},
            {"steps": 30, "guidance": 5.0, "type": "filter_sweep"},
        ]
        
        for i, params in enumerate(test_params):
            print(f"\nTest {i+1}: steps={params['steps']}, guidance={params['guidance']}, type={params['type']}")
            
            try:
                test_spec = generator.generate_transition(
                    song_a_path=song_a_path,
                    song_b_path=song_b_path,
                    transition_length=2.0,
                    tempo=120.0,
                    transition_type=params['type'],
                    num_inference_steps=params['steps'],
                    guidance_scale=params['guidance'],
                    seed=42
                )
                
                test_audio = mel_to_audio(test_spec, sr=22050, n_fft=1024, hop_length=256, n_iter=32)
                test_rms = torch.sqrt(torch.mean(test_audio**2))
                
                print(f"  Spec range: [{test_spec.min():.4f}, {test_spec.max():.4f}]")
                print(f"  Audio RMS: {test_rms:.4f}")
                
                # Save this test
                test_path = f'debug_test_{i+1}_transition.wav'
                amplified_test = test_audio * 10.0
                amplified_test = torch.clamp(amplified_test, -1.0, 1.0)
                torchaudio.save(test_path, amplified_test.unsqueeze(0), 22050)
                print(f"  Saved: {test_path}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Step 7: Compare with training data
        print("\n" + "="*60)
        print("STEP 7: COMPARISON WITH TRAINING DATA")
        print("="*60)
        
        try:
            # Try to load a training sample for comparison
            import pandas as pd
            metadata = pd.read_csv('metadata.csv')
            
            if len(metadata) > 0:
                sample_path = metadata.iloc[0]['path']
                target_path = Path(sample_path) / 'target.wav'
                
                if target_path.exists():
                    target_audio, _ = torchaudio.load(target_path)
                    target_rms = torch.sqrt(torch.mean(target_audio**2))
                    
                    print(f"Training target RMS: {target_rms:.4f}")
                    print(f"Generated audio RMS: {audio_energy:.4f}")
                    print(f"Ratio (generated/target): {audio_energy/target_rms:.4f}")
                    
                    if audio_energy / target_rms < 0.01:
                        print("🚨 PROBLEM: Generated audio is 100x quieter than training data!")
                    elif audio_energy / target_rms < 0.1:
                        print("⚠️  Generated audio is significantly quieter than training data")
                    else:
                        print("✓ Generated audio level is reasonable compared to training data")
                else:
                    print("Could not find training target for comparison")
            else:
                print("No metadata available for comparison")
                
        except Exception as e:
            print(f"Could not load training data for comparison: {e}")
        
        print("\n" + "="*60)
        print("DIAGNOSIS COMPLETE")
        print("="*60)
        
        print("\nKey findings:")
        print(f"- Model generates spectrograms with range [{transition_spec.min():.4f}, {transition_spec.max():.4f}]")
        print(f"- Generated audio RMS: {audio_energy:.4f}")
        print(f"- Check the amplified audio files: debug_comprehensive_transition.wav, debug_test_*_transition.wav")
        print(f"- Check the analysis plots: debug_comprehensive_analysis.png")
        
        if audio_energy < 1e-6:
            print("\n🔇 SILENT OUTPUT DIAGNOSIS:")
            print("1. The model IS generating content (spectrogram has variation)")
            print("2. The issue is likely in the spectrogram-to-audio conversion")
            print("3. Possible causes:")
            print("   - Mel-to-linear conversion scaling issues")
            print("   - Griffin-Lim reconstruction problems")
            print("   - Model trained on different audio normalization")
            print("4. Try listening to the amplified versions (10x gain)")
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_model_output()
