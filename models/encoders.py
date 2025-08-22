import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoConfig,
    Wav2Vec2Model, Wav2Vec2Config,
    ViTModel, ViTConfig
)
from typing import Dict, Tuple


class TextEncoder(nn.Module):
    """
    Advanced text encoder with adapter support
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(config.text_model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Adapter layers for few-shot learning
        self.adapter = AdapterLayer(
            self.hidden_size, 
            config.adapter_size
        ) if hasattr(config, 'adapter_size') else None
        
        # Prompt tuning embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(config.prompt_length, self.hidden_size)
        ) if hasattr(config, 'prompt_length') else None
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_size, config.fusion_hidden_size)
        self.dropout = nn.Dropout(config.fusion_dropout)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        use_adapter: bool = False,
        use_prompt: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = input_ids.size(0)
        
        # Add prompt tokens if enabled
        if use_prompt and self.prompt_embeddings is not None:
            prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            
            # Get input embeddings
            input_embeds = self.model.embeddings.word_embeddings(input_ids)
            
            # Concatenate prompt and input embeddings
            input_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            
            # Extend attention mask
            prompt_mask = torch.ones(
                batch_size, self.config.prompt_length,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            
            # Forward pass with embeddings
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask
            )
        else:
            # Standard forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Apply adapter if enabled
        if use_adapter and self.adapter is not None:
            sequence_output = self.adapter(sequence_output)
        
        # Pool sequence output (CLS token or mean pooling)
        if hasattr(self.model.config, 'model_type') and 'bert' in self.model.config.model_type:
            pooled_output = sequence_output[:, 0]  # CLS token
        else:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        
        # Project to fusion dimension
        projected = self.projection(pooled_output)
        projected = self.dropout(projected)
        
        return {
            'features': projected,
            'sequence_output': sequence_output,
            'attention_mask': attention_mask
        }


class AudioEncoder(nn.Module):
    """
    Advanced audio encoder with wav2vec2
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pre-trained wav2vec2
        self.model = Wav2Vec2Model.from_pretrained(config.audio_model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Adapter layers
        self.adapter = AdapterLayer(
            self.hidden_size, 
            config.adapter_size
        ) if hasattr(config, 'adapter_size') else None
        
        # Temporal attention for sequence modeling
        self.temporal_attention = nn.MultiheadAttention(
            self.hidden_size, 
            num_heads=8, 
            dropout=config.fusion_dropout,
            batch_first=True
        )
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_size, config.fusion_hidden_size)
        self.dropout = nn.Dropout(config.fusion_dropout)
        
    def forward(
        self, 
        waveform: torch.Tensor,
        use_adapter: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # Extract features with wav2vec2
        outputs = self.model(waveform)
        sequence_output = outputs.last_hidden_state
        
        # Apply adapter if enabled
        if use_adapter and self.adapter is not None:
            sequence_output = self.adapter(sequence_output)
        
        # Apply temporal attention
        attended_output, attention_weights = self.temporal_attention(
            sequence_output, sequence_output, sequence_output
        )
        
        # Global average pooling
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Project to fusion dimension
        projected = self.projection(pooled_output)
        projected = self.dropout(projected)
        
        return {
            'features': projected,
            'sequence_output': attended_output,
            'attention_weights': attention_weights
        }


class VideoEncoder(nn.Module):
    """
    Advanced video encoder with ViT and facial feature extraction
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision Transformer for frame features
        self.vit = ViTModel.from_pretrained(config.video_model_name)
        self.hidden_size = self.vit.config.hidden_size
        
        # Temporal modeling with LSTM
        self.temporal_lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.fusion_dropout
        )
        
        # Facial landmark attention
        self.facial_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            dropout=config.fusion_dropout,
            batch_first=True
        )
        
        # Adapter layers
        self.adapter = AdapterLayer(
            self.hidden_size, 
            config.adapter_size
        ) if hasattr(config, 'adapter_size') else None
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_size, config.fusion_hidden_size)
        self.dropout = nn.Dropout(config.fusion_dropout)
        
    def forward(
        self, 
        video_frames: torch.Tensor,
        use_adapter: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, num_frames, channels, height, width = video_frames.shape
        
        # Reshape for ViT processing
        frames_flat = video_frames.view(-1, channels, height, width)
        
        # Extract frame features with ViT
        vit_outputs = self.vit(pixel_values=frames_flat)
        frame_features = vit_outputs.last_hidden_state[:, 0]  # CLS token
        
        # Reshape back to sequence
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Apply adapter if enabled
        if use_adapter and self.adapter is not None:
            frame_features = self.adapter(frame_features)
        
        # Temporal modeling with LSTM
        lstm_output, (hidden, cell) = self.temporal_lstm(frame_features)
        
        # Apply facial attention
        attended_output, attention_weights = self.facial_attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Global average pooling
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Project to fusion dimension
        projected = self.projection(pooled_output)
        projected = self.dropout(projected)
        
        return {
            'features': projected,
            'sequence_output': attended_output,
            'attention_weights': attention_weights
        }


class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning
    """
    def __init__(self, hidden_size: int, adapter_size: int):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize with small weights
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return residual + x


class ModalityDropout(nn.Module):
    """
    Modality dropout for robustness training
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if not training:
            return text_features, audio_features, video_features
        
        batch_size = text_features.size(0)
        
        # Random modality dropout
        text_mask = torch.rand(batch_size, 1, device=text_features.device) > self.dropout_rate
        audio_mask = torch.rand(batch_size, 1, device=audio_features.device) > self.dropout_rate
        video_mask = torch.rand(batch_size, 1, device=video_features.device) > self.dropout_rate
        
        # Ensure at least one modality is kept
        all_dropped = ~(text_mask | audio_mask | video_mask).squeeze()
        if all_dropped.any():
            # Randomly keep one modality for samples where all were dropped
            random_choice = torch.randint(0, 3, (all_dropped.sum(),), device=text_features.device)
            text_mask[all_dropped] = (random_choice == 0).unsqueeze(1)
            audio_mask[all_dropped] = (random_choice == 1).unsqueeze(1)
            video_mask[all_dropped] = (random_choice == 2).unsqueeze(1)
        
        # Apply masks
        text_features = text_features * text_mask.float()
        audio_features = audio_features * audio_mask.float()
        video_features = video_features * video_mask.float()
        
        return text_features, audio_features, video_features
