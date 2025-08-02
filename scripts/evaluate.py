#!/usr/bin/env python3
"""
Evaluation script for DJNet Diffusion Model
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.generator import DJNetGenerator
from src.data.dataset import create_dataloaders
from src.training.trainer import load_config
from src.utils.audio_utils import compute_audio_features


def compute_transition_metrics(generated_audio, target_audio, sample_rate):
    """Compute metrics comparing generated and target transitions."""
    
    # Convert to numpy
    if isinstance(generated_audio, torch.Tensor):
        generated_audio = generated_audio.detach().cpu().numpy()
    if isinstance(target_audio, torch.Tensor):
        target_audio = target_audio.detach().cpu().numpy()
    
    # Ensure same length
    min_length = min(len(generated_audio), len(target_audio))
    generated_audio = generated_audio[:min_length]
    target_audio = target_audio[:min_length]
    
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = float(np.mean((generated_audio - target_audio) ** 2))
    
    # Signal-to-Noise Ratio
    signal_power = np.mean(target_audio ** 2)
    noise_power = np.mean((generated_audio - target_audio) ** 2)
    metrics['snr'] = float(10 * np.log10(signal_power / (noise_power + 1e-8)))
    
    # Spectral Distance (simplified)
    from scipy import signal
    f_gen, psd_gen = signal.welch(generated_audio, fs=sample_rate)
    f_target, psd_target = signal.welch(target_audio, fs=sample_rate)
    
    # Log spectral distance
    log_psd_gen = np.log(psd_gen + 1e-8)
    log_psd_target = np.log(psd_target + 1e-8)
    metrics['spectral_distance'] = float(np.mean((log_psd_gen - log_psd_target) ** 2))
    
    return metrics


def evaluate_model(checkpoint_path, config_path, model_config_path, num_samples=100):
    """Evaluate trained model on test set."""
    
    # Load configurations
    training_config = load_config(config_path)
    model_config = load_config(model_config_path)
    config = {**model_config, **training_config}
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize generator
    generator = DJNetGenerator(checkpoint_path, device)
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(config)
    
    # Evaluation metrics
    all_metrics = []
    
    print(f"Evaluating model on {num_samples} test samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            if i >= num_samples // config['training']['batch_size']:
                break
            
            for j in range(min(batch['target_transition_spectrogram'].shape[0], 
                              num_samples - len(all_metrics))):
                
                try:
                    # Extract single sample
                    sample_batch = {
                        k: v[j:j+1] if isinstance(v, torch.Tensor) else [v[j]] 
                        for k, v in batch.items()
                    }
                    
                    # Generate transition
                    generated_spec = generator.generate_transition(
                        song_a_path="dummy",  # We'll use the spectrograms directly
                        song_b_path="dummy",
                        transition_length=float(sample_batch['transition_length'][0]),
                        tempo=float(sample_batch['avg_tempo'][0]),
                        transition_type=sample_batch['transition_type'][0],
                        num_inference_steps=20,  # Faster for evaluation
                        seed=i * 1000 + j
                    )
                    
                    # Convert spectrograms to audio for comparison
                    from src.utils.audio_utils import spectrogram_to_audio
                    
                    generated_audio = spectrogram_to_audio(generated_spec, config)
                    target_audio = spectrogram_to_audio(
                        sample_batch['target_transition_spectrogram'], config
                    )
                    
                    # Compute metrics
                    metrics = compute_transition_metrics(
                        generated_audio, target_audio, config['audio']['sample_rate']
                    )
                    
                    # Add metadata
                    metrics.update({
                        'transition_id': sample_batch['transition_id'][0],
                        'transition_type': sample_batch['transition_type'][0],
                        'transition_length': float(sample_batch['transition_length'][0]),
                        'avg_tempo': float(sample_batch['avg_tempo'][0])
                    })
                    
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"Error evaluating sample {len(all_metrics)}: {str(e)}")
                    continue
                
                if len(all_metrics) >= num_samples:
                    break
            
            if len(all_metrics) >= num_samples:
                break
    
    # Convert to DataFrame and compute statistics
    df = pd.DataFrame(all_metrics)
    
    # Overall statistics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric in ['mse', 'snr', 'spectral_distance']:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Statistics by transition type
    print("\nBY TRANSITION TYPE:")
    print("-" * 30)
    for t_type in df['transition_type'].unique():
        subset = df[df['transition_type'] == t_type]
        print(f"\n{t_type.upper()}:")
        for metric in ['mse', 'snr', 'spectral_distance']:
            mean_val = subset[metric].mean()
            print(f"  {metric}: {mean_val:.4f}")
    
    # Save detailed results
    results_path = Path(checkpoint_path).parent / 'evaluation_results.csv'
    df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate DJNet Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of test samples to evaluate')
    
    args = parser.parse_args()
    
    print(f"Evaluating model: {args.checkpoint}")
    print(f"Number of samples: {args.num_samples}")
    
    try:
        results = evaluate_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            model_config_path=args.model_config,
            num_samples=args.num_samples
        )
        print("\n Evaluation completed successfully!")
        
    except Exception as e:
        print(f" Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
