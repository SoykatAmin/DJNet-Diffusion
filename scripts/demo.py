#!/usr/bin/env python3
"""
Demo script for DJNet Diffusion Model
Interactive generation with different parameters
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.generator import DJNetGenerator


def interactive_demo(checkpoint_path):
    """Interactive demo for generating transitions."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DJNet model on {device}...")
    
    # Initialize generator
    generator = DJNetGenerator(checkpoint_path, device)
    
    print("\n" + "="*60)
    print(" DJNet Interactive Demo ")
    print("="*60)
    print("Generate seamless transitions between your music tracks!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Available transition types
    transition_types = ['echo_fade', 'bass_swap_eq', 'exp_fade', 'linear_fade', 'hard_cut', 'filter_sweep']
    
    while True:
        try:
            print("\n" + "-"*40)
            
            # Get input files
            song_a = input("Enter path to Song A: ").strip()
            if song_a.lower() in ['quit', 'exit']:
                break
            
            if not Path(song_a).exists():
                print(" Song A not found!")
                continue
            
            song_b = input("Enter path to Song B: ").strip()
            if song_b.lower() in ['quit', 'exit']:
                break
                
            if not Path(song_b).exists():
                print(" Song B not found!")
                continue
            
            # Get output path
            output = input("Enter output path (or press Enter for auto): ").strip()
            if not output:
                output = f"generated_transition_{Path(song_a).stem}_to_{Path(song_b).stem}.wav"
            
            # Get parameters
            try:
                length = input("Transition length in seconds (default: 8.0): ").strip()
                length = float(length) if length else 8.0
                
                tempo = input("Target tempo/BPM (default: 120.0): ").strip()
                tempo = float(tempo) if tempo else 120.0
                
                print(f"\nAvailable transition types: {', '.join(transition_types)}")
                t_type = input("Transition type (default: linear_fade): ").strip()
                t_type = t_type if t_type in transition_types else 'linear_fade'
                
                steps = input("Number of denoising steps (default: 50): ").strip()
                steps = int(steps) if steps else 50
                
                guidance = input("Guidance scale (default: 1.0): ").strip()
                guidance = float(guidance) if guidance else 1.0
                
                seed = input("Random seed (optional): ").strip()
                seed = int(seed) if seed else None
                
            except ValueError:
                print(" Invalid parameter values!")
                continue
            
            # Generate transition
            print(f"\n Generating transition...")
            print(f"   Song A: {Path(song_a).name}")
            print(f"   Song B: {Path(song_b).name}")
            print(f"   Length: {length}s")
            print(f"   Tempo: {tempo} BPM")
            print(f"   Type: {t_type}")
            print(f"   Steps: {steps}")
            
            try:
                output_path = generator.generate_transition_audio(
                    song_a_path=song_a,
                    song_b_path=song_b,
                    output_path=output,
                    transition_length=length,
                    tempo=tempo,
                    transition_type=t_type,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    add_context=True,
                    crossfade_duration=0.5
                )
                
                print(f" Transition generated: {output_path}")
                
                # Ask if user wants to generate another
                another = input("\nGenerate another transition? (y/n): ").strip().lower()
                if another not in ['y', 'yes']:
                    break
                    
            except Exception as e:
                print(f" Error generating transition: {str(e)}")
                continue
                
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f" Unexpected error: {str(e)}")
            continue
    
    print("\n Thanks for using DJNet! ")


def batch_demo(checkpoint_path, song_pairs_file, output_dir):
    """Batch generation demo."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DJNet model on {device}...")
    
    # Initialize generator
    generator = DJNetGenerator(checkpoint_path, device)
    
    # Read song pairs
    song_pairs = []
    with open(song_pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    song_pairs.append((parts[0].strip(), parts[1].strip()))
    
    print(f"Generating transitions for {len(song_pairs)} song pairs...")
    
    # Generate transitions
    output_paths = generator.batch_generate(
        song_pairs=song_pairs,
        output_dir=output_dir,
        transition_length=8.0,
        tempo=120.0,
        transition_type='linear_fade',
        num_inference_steps=50,
        add_context=True
    )
    
    print(f"\n Generated {len([p for p in output_paths if p])} transitions")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DJNet Demo - Generate Music Transitions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], 
                       default='interactive',
                       help='Demo mode: interactive or batch')
    parser.add_argument('--song_pairs', type=str, default=None,
                       help='CSV file with song pairs for batch mode')
    parser.add_argument('--output_dir', type=str, default='demo_outputs',
                       help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f" Checkpoint not found: {args.checkpoint}")
        return
    
    if args.mode == 'interactive':
        interactive_demo(args.checkpoint)
    elif args.mode == 'batch':
        if not args.song_pairs:
            print(" --song_pairs required for batch mode")
            return
        if not Path(args.song_pairs).exists():
            print(f" Song pairs file not found: {args.song_pairs}")
            return
        batch_demo(args.checkpoint, args.song_pairs, args.output_dir)


if __name__ == '__main__':
    main()
