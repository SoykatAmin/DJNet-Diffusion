import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to handle tensors of different sizes."""
    if not batch:
        return {}
    
    # Debug: Print that custom collate is being used
    # print(f"Custom collate function called with batch size: {len(batch)}")
    
    # Instead of finding max, use fixed expected dimensions
    # This should be consistent since we now force fixed dimensions in the dataset
    
    # Initialize lists for each field
    collated = {
        'preceding_spectrogram': [],
        'following_spectrogram': [],
        'target_transition_spectrogram': [],
        'transition_type': [],
        'transition_length': [],
        'avg_tempo': [],
        'transition_id': []
    }
    
    # Check if all spectrograms have the same shape
    first_item = batch[0]
    expected_prec_shape = first_item['preceding_spectrogram'].shape
    expected_foll_shape = first_item['following_spectrogram'].shape  
    expected_trans_shape = first_item['target_transition_spectrogram'].shape
    
    for item in batch:
        # Verify shapes match expected dimensions
        for spec_key, expected_shape in [
            ('preceding_spectrogram', expected_prec_shape),
            ('following_spectrogram', expected_foll_shape),
            ('target_transition_spectrogram', expected_trans_shape)
        ]:
            spec = item[spec_key]
            
            if spec.shape != expected_shape:
                print(f"Warning: {spec_key} shape mismatch. Expected {expected_shape}, got {spec.shape}")
                # Force reshape to expected dimensions
                if spec.shape[-1] != expected_shape[-1]:
                    if spec.shape[-1] < expected_shape[-1]:
                        padding = expected_shape[-1] - spec.shape[-1]
                        spec = F.pad(spec, (0, padding), mode='constant', value=0)
                    else:
                        spec = spec[..., :expected_shape[-1]]
            
            # Ensure tensor is detached and contiguous
            spec = spec.detach().contiguous()
            collated[spec_key].append(spec)
        
        # Add other fields as-is
        collated['transition_type'].append(item['transition_type'])
        collated['transition_length'].append(float(item['transition_length']))
        collated['avg_tempo'].append(float(item['avg_tempo']))
        collated['transition_id'].append(item['transition_id'])
    
    # Stack tensors
    try:
        for spec_key in ['preceding_spectrogram', 'following_spectrogram', 'target_transition_spectrogram']:
            collated[spec_key] = torch.stack(collated[spec_key], dim=0)
    except RuntimeError as e:
        print(f"Error stacking tensors for {spec_key}: {e}")
        print(f"Tensor shapes: {[t.shape for t in collated[spec_key]]}")
        raise
    
    # Convert scalar lists to tensors
    collated['transition_length'] = torch.tensor(collated['transition_length'], dtype=torch.float32)
    collated['avg_tempo'] = torch.tensor(collated['avg_tempo'], dtype=torch.float32)
    
    return collated


class SpectrogramProcessor:
    """Handles audio to spectrogram conversion and normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.sample_rate = config['audio']['sample_rate']
        self.n_mels = config['audio']['n_mels']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.win_length = config['audio']['win_length']
        self.spec_min = config['audio']['spec_min']
        self.spec_max = config['audio']['spec_max']
        self.normalize = config['audio']['normalize_spectrograms']
        
        # Create mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
    
    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(audio)
        
        # Convert to dB
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        if self.normalize:
            # Normalize to [-1, 1] range
            mel_spec_db = torch.clamp(mel_spec_db, self.spec_min, self.spec_max)
            mel_spec_db = 2.0 * (mel_spec_db - self.spec_min) / (self.spec_max - self.spec_min) - 1.0
        
        return mel_spec_db
    
    def load_audio(self, file_path: str, duration: Optional[float] = None) -> torch.Tensor:
        """Load and resample audio file."""
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Trim or pad to desired duration
        if duration is not None:
            target_length = int(duration * self.sample_rate)
            current_length = audio.shape[1]
            
            if current_length > target_length:
                # Trim from center
                start_idx = (current_length - target_length) // 2
                audio = audio[:, start_idx:start_idx + target_length]
            elif current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio.squeeze(0)  # Remove channel dimension


class DJNetTransitionDataset(Dataset):
    """Dataset for loading DJ transition data with context."""
    
    def __init__(
        self,
        metadata_path: str,
        data_root: str,
        config: Dict[str, Any],
        split: str = 'train',
        transform=None
    ):
        self.data_root = Path(data_root)
        self.config = config
        self.transform = transform
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Create train/val/test split
        self.metadata = self.metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_split = config['data']['train_split']
        val_split = config['data']['val_split']
        
        n_total = len(self.metadata)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        if split == 'train':
            self.metadata = self.metadata[:n_train]
        elif split == 'val':
            self.metadata = self.metadata[n_train:n_train + n_val]
        elif split == 'test':
            self.metadata = self.metadata[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Initialize spectrogram processor
        self.spec_processor = SpectrogramProcessor(config)
        
        # Audio parameters
        self.context_duration = config['audio']['context_duration']
        self.max_transition_duration = config['audio']['max_transition_duration']
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _get_fixed_spectrogram_size(self, duration: float) -> Tuple[int, int]:
        """Calculate the size of spectrogram for given duration."""
        n_frames = int(duration * self.spec_processor.sample_rate / self.spec_processor.hop_length) + 1
        return self.spec_processor.n_mels, n_frames
    
    def _pad_or_crop_spectrogram(self, spec: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Pad or crop spectrogram to target number of frames."""
        current_frames = spec.shape[-1]
        
        if current_frames > target_frames:
            # Crop from center
            start_idx = (current_frames - target_frames) // 2
            spec = spec[..., start_idx:start_idx + target_frames]
        elif current_frames < target_frames:
            # Pad with zeros
            padding = target_frames - current_frames
            spec = torch.nn.functional.pad(spec, (0, padding))
        
        return spec
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        
        transition_id = row['id']
        transition_path = self.data_root / transition_id
        transition_type = row['transition_type']
        transition_length = row['transition_length_sec']
        avg_tempo = row['avg_tempo']
        
        try:
            # Load conditioning information
            conditioning_file = transition_path / 'conditioning.json'
            conditioning = {}
            if conditioning_file.exists():
                import json
                with open(conditioning_file, 'r') as f:
                    conditioning = json.load(f)
            
            # Load source audio files (full segments)
            source_a_audio = self.spec_processor.load_audio(
                str(transition_path / 'source_a.wav')
            )
            
            source_b_audio = self.spec_processor.load_audio(
                str(transition_path / 'source_b.wav')
            )
            
            # Load target transition with fixed duration to ensure consistent spectrograms
            fixed_duration = min(transition_length, self.max_transition_duration)
            transition_audio = self.spec_processor.load_audio(
                str(transition_path / 'target.wav'),
                duration=fixed_duration
            )
            
            # Extract context segments around transition points
            # Use conditioning info if available, otherwise use defaults
            start_pos_a = conditioning.get('start_position_a_sec', len(source_a_audio) / self.spec_processor.sample_rate - self.context_duration)
            start_pos_b = conditioning.get('start_position_b_sec', 0.0)
            
            # Extract preceding context (before transition in source A) with fixed duration
            context_samples = int(self.context_duration * self.spec_processor.sample_rate)
            start_sample_a = max(0, int((start_pos_a - self.context_duration) * self.spec_processor.sample_rate))
            end_sample_a = start_sample_a + context_samples
            
            if end_sample_a <= len(source_a_audio):
                preceding_audio = source_a_audio[start_sample_a:end_sample_a]
            else:
                # Fallback: take last context_duration seconds
                preceding_audio = source_a_audio[-context_samples:] if len(source_a_audio) >= context_samples else source_a_audio
                # Pad if too short
                if len(preceding_audio) < context_samples:
                    padding = context_samples - len(preceding_audio)
                    preceding_audio = torch.nn.functional.pad(preceding_audio, (padding, 0))
            
            # Extract following context (after transition start in source B) with fixed duration
            start_sample_b = int(start_pos_b * self.spec_processor.sample_rate)
            end_sample_b = start_sample_b + context_samples
            
            if end_sample_b <= len(source_b_audio):
                following_audio = source_b_audio[start_sample_b:end_sample_b]
            else:
                # Fallback: take first context_duration seconds
                following_audio = source_b_audio[:context_samples] if len(source_b_audio) >= context_samples else source_b_audio
                # Pad if too short
                if len(following_audio) < context_samples:
                    padding = context_samples - len(following_audio)
                    following_audio = torch.nn.functional.pad(following_audio, (0, padding))
            
            # Ensure all audio segments have exactly the same length
            preceding_audio = preceding_audio[:context_samples] if len(preceding_audio) > context_samples else preceding_audio
            following_audio = following_audio[:context_samples] if len(following_audio) > context_samples else following_audio
            
            # Convert to spectrograms
            preceding_spec = self.spec_processor.audio_to_spectrogram(preceding_audio)
            transition_spec = self.spec_processor.audio_to_spectrogram(transition_audio)
            following_spec = self.spec_processor.audio_to_spectrogram(following_audio)
            
            # Calculate expected spectrogram size for fixed duration
            expected_frames = int(fixed_duration * self.spec_processor.sample_rate / self.spec_processor.hop_length) + 1
            context_frames = int(self.context_duration * self.spec_processor.sample_rate / self.spec_processor.hop_length) + 1
            
            # Force all spectrograms to have exact expected dimensions
            preceding_spec = self._pad_or_crop_spectrogram(preceding_spec, context_frames)
            following_spec = self._pad_or_crop_spectrogram(following_spec, context_frames)
            transition_spec = self._pad_or_crop_spectrogram(transition_spec, expected_frames)
            
            # Add channel dimension if needed
            if preceding_spec.dim() == 2:
                preceding_spec = preceding_spec.unsqueeze(0)
            if following_spec.dim() == 2:
                following_spec = following_spec.unsqueeze(0)
            if transition_spec.dim() == 2:
                transition_spec = transition_spec.unsqueeze(0)
            
            sample = {
                'preceding_spectrogram': preceding_spec.float(),
                'following_spectrogram': following_spec.float(),
                'target_transition_spectrogram': transition_spec.float(),
                'transition_type': transition_type,
                'transition_length': float(transition_length),
                'avg_tempo': float(avg_tempo),
                'transition_id': transition_id
            }
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx} ({transition_id}): {str(e)}")
            # Return a dummy sample or skip
            return self.__getitem__((idx + 1) % len(self.metadata))


def create_dataloaders(config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, validation, and test dataloaders."""
    
    # Create datasets
    train_dataset = DJNetTransitionDataset(
        metadata_path=config['data']['metadata_path'],
        data_root=config['data']['data_root'],
        config=config,
        split='train'
    )
    
    val_dataset = DJNetTransitionDataset(
        metadata_path=config['data']['metadata_path'],
        data_root=config['data']['data_root'],
        config=config,
        split='val'
    )
    
    test_dataset = DJNetTransitionDataset(
        metadata_path=config['data']['metadata_path'],
        data_root=config['data']['data_root'],
        config=config,
        split='test'
    )
    
    # Create dataloaders with num_workers=0 to avoid multiprocessing issues
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=config['data']['pin_memory'],
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=config['data']['pin_memory'],
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=config['data']['pin_memory'],
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader
