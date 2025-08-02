#!/bin/bash

# Quick setup script for DJNet-Diffusion

echo "🎵 Setting up DJNet-Diffusion..."

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs
mkdir -p demo_outputs

# Check if dataset path exists in metadata.csv
echo "Checking dataset paths..."
if [ ! -d "./output/djnet_dataset_20k" ]; then
    echo "Dataset directory not found: ./output/djnet_dataset_20k"
    echo "Please ensure your dataset is in the correct location or update configs/training_config.yaml"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Ensure your dataset is in ./output/djnet_dataset_20k/ (or update the config)"
echo "2. Start training: python scripts/train.py"
echo "3. Generate transitions: python scripts/inference.py --checkpoint path/to/model.pth --song_a song1.wav --song_b song2.wav --output transition.wav"
echo "4. Try the demo: python scripts/demo.py --checkpoint path/to/model.pth"
