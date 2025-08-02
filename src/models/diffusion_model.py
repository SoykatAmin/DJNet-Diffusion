import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any


class ConditioningEncoder(nn.Module):
    """Encodes conditioning information (tempo, transition type, length) into embeddings."""
    
    def __init__(
        self,
        tempo_embed_dim: int = 64,
        type_embed_dim: int = 64,
        length_embed_dim: int = 64,
        cross_attention_dim: int = 256,
        transition_types: list = None
    ):
        super().__init__()
        
        if transition_types is None:
            transition_types = ['echo_fade', 'bass_swap_eq', 'exp_fade', 'linear_fade', 
                               'hard_cut', 'filter_sweep']
        
        self.transition_types = transition_types
        self.num_types = len(transition_types)
        
        # Embedding layers
        self.tempo_projection = nn.Linear(1, tempo_embed_dim)
        self.type_embedding = nn.Embedding(self.num_types, type_embed_dim)
        self.length_projection = nn.Linear(1, length_embed_dim)
        
        # Combine all embeddings
        total_embed_dim = tempo_embed_dim + type_embed_dim + length_embed_dim
        self.output_projection = nn.Linear(total_embed_dim, cross_attention_dim)
        
        # Create type mapping
        self.type_to_idx = {t: i for i, t in enumerate(transition_types)}
    
    def encode_transition_type(self, transition_types: list) -> torch.Tensor:
        """Convert transition type strings to indices."""
        indices = []
        for t in transition_types:
            if t in self.type_to_idx:
                indices.append(self.type_to_idx[t])
            else:
                # Default to first type if unknown
                indices.append(0)
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(
        self, 
        tempo: torch.Tensor,
        transition_types: list,
        transition_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tempo: (batch_size,) tensor of tempo values
            transition_types: list of transition type strings
            transition_lengths: (batch_size,) tensor of transition lengths
        
        Returns:
            (batch_size, cross_attention_dim) conditioning embeddings
        """
        batch_size = tempo.shape[0]
        device = tempo.device
        
        # Encode tempo
        tempo_emb = self.tempo_projection(tempo.unsqueeze(-1))  # (batch_size, tempo_embed_dim)
        
        # Encode transition type
        type_indices = self.encode_transition_type(transition_types).to(device)
        type_emb = self.type_embedding(type_indices)  # (batch_size, type_embed_dim)
        
        # Encode transition length
        length_emb = self.length_projection(transition_lengths.unsqueeze(-1))  # (batch_size, length_embed_dim)
        
        # Concatenate all embeddings
        combined = torch.cat([tempo_emb, type_emb, length_emb], dim=-1)  # (batch_size, total_embed_dim)
        
        # Project to cross attention dimension
        output = self.output_projection(combined)  # (batch_size, cross_attention_dim)
        
        return output.unsqueeze(1)  # (batch_size, 1, cross_attention_dim) for cross attention


class DJNetDiffusionModel(nn.Module):
    """Main DJNet diffusion model combining UNet with conditioning."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Initialize conditioning encoder
        self.conditioning_encoder = ConditioningEncoder(
            tempo_embed_dim=config['conditioning']['tempo_embed_dim'],
            type_embed_dim=config['conditioning']['type_embed_dim'],
            length_embed_dim=config['conditioning']['length_embed_dim'],
            cross_attention_dim=config['model']['cross_attention_dim']
        )
        
        # Initialize UNet
        self.unet = UNet2DConditionModel(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            down_block_types=config['model']['down_block_types'],
            up_block_types=config['model']['up_block_types'],
            block_out_channels=config['model']['block_out_channels'],
            layers_per_block=config['model']['layers_per_block'],
            attention_head_dim=config['model']['attention_head_dim'],
            norm_num_groups=config['model']['norm_num_groups'],
            cross_attention_dim=config['model']['cross_attention_dim']
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        tempo: torch.Tensor,
        transition_types: list,
        transition_lengths: torch.Tensor,
        return_dict: bool = True
    ):
        """
        Args:
            sample: (batch_size, in_channels, height, width) input tensor
            timestep: (batch_size,) timestep tensor
            tempo: (batch_size,) tempo values
            transition_types: list of transition type strings
            transition_lengths: (batch_size,) transition length values
            return_dict: whether to return dict or tensor
        """
        # Encode conditioning information
        encoder_hidden_states = self.conditioning_encoder(
            tempo=tempo,
            transition_types=transition_types,
            transition_lengths=transition_lengths
        )
        
        # Forward through UNet
        unet_output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        
        if return_dict:
            return unet_output
        else:
            return unet_output.sample
