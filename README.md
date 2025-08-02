# DJNet-Diffusion: Generative Transitions for Seamless Music Mixing

A diffusion-based model for generating seamless transitions between music tracks, trained on labeled (context, target) pairs.

## Overview

DJNet-Diffusion learns to generate high-quality audio transitions between songs using a conditional diffusion model. Given the end of Song A and the start of Song B, the model generates a smooth transition that blends them seamlessly.

## Features

- Conditional diffusion model using UNet2D architecture
- Mel-spectrogram based audio representation
- Support for multiple transition types (fade, bass_swap, filter_sweep, etc.)
- Tempo and transition length conditioning
- Comprehensive training and inference pipeline

## Dataset Structure

The model expects transitions organized as:
```
transition_XXXXX/
├── preceding_audio.wav    # End of Song A
├── transition_audio.wav   # The actual transition
└── following_audio.wav    # Start of Song B
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python scripts/train.py --config configs/training_config.yaml
```

3. Generate transitions:
```bash
python scripts/inference.py --checkpoint path/to/model.pth --input_a song_a.wav --input_b song_b.wav
```

## Model Architecture

- **Backbone**: UNet2DConditionModel from diffusers
- **Scheduler**: DDPMScheduler for training and inference
- **Audio Representation**: Mel-spectrograms (128 mel bins, various time lengths)
- **Conditioning**: Concatenated context spectrograms + tempo/length embeddings

## Training

The model is trained to denoise corrupted transitions given musical context:
- Input: [preceding_spec, following_spec, noisy_transition] + timestep
- Target: Original noise that was added
- Loss: MSE between predicted and actual noise

## License

MIT License
