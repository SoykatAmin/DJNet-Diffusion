#!/usr/bin/env python3
"""
Inference script for DJNet Diffusion Model
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.generator import DJNetGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate transitions with DJNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--song_a', type=str, required=True,
                       help='Path to first song (Song A)')
    parser.add_argument('--song_b', type=str, required=True,
                       help='Path to second song (Song B)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for generated transition')
    parser.add_argument('--transition_length', type=float, default=8.0,
                       help='Length of transition in seconds')
    parser.add_argument('--tempo', type=float, default=120.0,
                       help='Target tempo for transition')
    parser.add_argument('--transition_type', type=str, default='linear_fade',
                       choices=['echo_fade', 'bass_swap_eq', 'exp_fade', 'linear_fade', 
                               'hard_cut', 'filter_sweep'],
                       help='Type of transition to generate')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='Guidance scale for conditional generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_context', action='store_true',
                       help='Generate only transition without context')
    parser.add_argument('--crossfade_duration', type=float, default=0.5,
                       help='Duration of crossfade with context in seconds')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Loading model from: {args.checkpoint}")
    
    # Initialize generator
    generator = DJNetGenerator(args.checkpoint, device)
    
    # Generate transition
    print(f"Generating transition between:")
    print(f"  Song A: {args.song_a}")
    print(f"  Song B: {args.song_b}")
    print(f"  Length: {args.transition_length}s")
    print(f"  Tempo: {args.tempo} BPM")
    print(f"  Type: {args.transition_type}")
    print(f"  Steps: {args.num_steps}")
    
    try:
        output_path = generator.generate_transition_audio(
            song_a_path=args.song_a,
            song_b_path=args.song_b,
            output_path=args.output,
            transition_length=args.transition_length,
            tempo=args.tempo,
            transition_type=args.transition_type,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            add_context=not args.no_context,
            crossfade_duration=args.crossfade_duration
        )
        
        print(f" Transition generated successfully: {output_path}")
        
    except Exception as e:
        print(f" Error generating transition: {str(e)}")
        raise


if __name__ == '__main__':
    main()
