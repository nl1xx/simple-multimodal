import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict


class EarlyFusion(nn.Module):
    """
    Simple early fusion by concatenation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input dimensions
        input_dim = config.fusion_hidden_size * 3  # text + audio + video
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(input_dim, config.fusion_hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_size * 2, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        
        # Concatenate features
        fused_features = torch.cat([text_features, audio_features, video_features], dim=-1)
        
        # Apply fusion layers
        output = self.fusion_layers(fused_features)
        
        return output


class LateFusion(nn.Module):
    """
    Late fusion with learned weights
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Individual classifiers for each modality
        self.text_classifier = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        self.audio_classifier = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        self.video_classifier = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        
        # Learned fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # Individual predictions
        text_logits = self.text_classifier(text_features)
        audio_logits = self.audio_classifier(audio_features)
        video_logits = self.video_classifier(video_features)
        
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted fusion
        fused_logits = (
            weights[0] * text_logits +
            weights[1] * audio_logits +
            weights[2] * video_logits
        )
        
        return {
            'fused_logits': fused_logits,
            'text_logits': text_logits,
            'audio_logits': audio_logits,
            'video_logits': video_logits,
            'fusion_weights': weights
        }


class MultimodalTransformer(nn.Module):
    """
    MulT: Multimodal Transformer with cross-modal attention
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Cross-modal transformers
        self.text_to_audio = CrossModalTransformer(config)
        self.text_to_video = CrossModalTransformer(config)
        self.audio_to_text = CrossModalTransformer(config)
        self.audio_to_video = CrossModalTransformer(config)
        self.video_to_text = CrossModalTransformer(config)
        self.video_to_audio = CrossModalTransformer(config)
        
        # Self-attention for each modality
        self.text_self_attn = nn.MultiheadAttention(
            config.fusion_hidden_size, config.fusion_num_heads,
            dropout=config.fusion_dropout, batch_first=True
        )
        self.audio_self_attn = nn.MultiheadAttention(
            config.fusion_hidden_size, config.fusion_num_heads,
            dropout=config.fusion_dropout, batch_first=True
        )
        self.video_self_attn = nn.MultiheadAttention(
            config.fusion_hidden_size, config.fusion_num_heads,
            dropout=config.fusion_dropout, batch_first=True
        )
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(config.fusion_hidden_size * 3, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = text_features.size(0)
        
        # Add sequence dimension if needed
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
            audio_features = audio_features.unsqueeze(1)
            video_features = video_features.unsqueeze(1)
        
        # Cross-modal attention
        text_from_audio = self.text_to_audio(text_features, audio_features)
        text_from_video = self.text_to_video(text_features, video_features)
        
        audio_from_text = self.audio_to_text(audio_features, text_features)
        audio_from_video = self.audio_to_video(audio_features, video_features)
        
        video_from_text = self.video_to_text(video_features, text_features)
        video_from_audio = self.video_to_audio(video_features, audio_features)
        
        # Combine cross-modal features
        enhanced_text = text_features + text_from_audio + text_from_video
        enhanced_audio = audio_features + audio_from_text + audio_from_video
        enhanced_video = video_features + video_from_text + video_from_audio
        
        # Self-attention
        text_attended, _ = self.text_self_attn(enhanced_text, enhanced_text, enhanced_text)
        audio_attended, _ = self.audio_self_attn(enhanced_audio, enhanced_audio, enhanced_audio)
        video_attended, _ = self.video_self_attn(enhanced_video, enhanced_video, enhanced_video)
        
        # Pool sequences
        text_pooled = torch.mean(text_attended, dim=1)
        audio_pooled = torch.mean(audio_attended, dim=1)
        video_pooled = torch.mean(video_attended, dim=1)
        
        # Final fusion
        fused_features = torch.cat([text_pooled, audio_pooled, video_pooled], dim=-1)
        output = self.final_fusion(fused_features)
        
        return {
            'fused_features': output,
            'text_features': text_pooled,
            'audio_features': audio_pooled,
            'video_features': video_pooled
        }


class CrossModalTransformer(nn.Module):
    """
    Cross-modal transformer block
    """
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.fusion_hidden_size, config.fusion_num_heads,
            dropout=config.fusion_dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.fusion_hidden_size)
        self.norm2 = nn.LayerNorm(config.fusion_hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_size * 4, config.fusion_hidden_size)
        )
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        attn_output, _ = self.attention(query, key_value, key_value)
        x = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class GraphFusion(nn.Module):
    """
    Graph-based multimodal fusion using GNN
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            GATConv(
                config.fusion_hidden_size, 
                config.graph_hidden_size,
                heads=4, 
                dropout=config.graph_dropout,
                concat=False
            )
            for _ in range(config.graph_num_layers)
        ])
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(3, config.fusion_hidden_size)  # text, audio, video
        
        # Final projection
        self.output_projection = nn.Linear(config.graph_hidden_size, config.fusion_hidden_size)
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size = text_features.size(0)
        device = text_features.device
        
        # Create graph data for each sample in batch
        batch_graphs = []
        
        for i in range(batch_size):
            # Node features: [text, audio, video]
            node_features = torch.stack([
                text_features[i], 
                audio_features[i], 
                video_features[i]
            ])
            
            # Add node type embeddings
            node_types = torch.tensor([0, 1, 2], device=device)
            type_embeds = self.node_type_embedding(node_types)
            node_features = node_features + type_embeds
            
            # Create fully connected graph (all nodes connected)
            edge_index = torch.tensor([
                [0, 0, 1, 1, 2, 2],  # source nodes
                [1, 2, 0, 2, 0, 1]   # target nodes
            ], device=device)
            
            # Create graph data
            graph_data = Data(x=node_features, edge_index=edge_index)
            batch_graphs.append(graph_data)
        
        # Batch graphs
        batched_graph = Batch.from_data_list(batch_graphs)
        
        # Apply GCN layers
        x = batched_graph.x
        for gcn_layer in self.gcn_layers:
            x = F.relu(gcn_layer(x, batched_graph.edge_index))
        
        # Global pooling for each graph in batch
        batch_indices = batched_graph.batch
        pooled_features = global_mean_pool(x, batch_indices)
        
        # Project to output dimension
        output = self.output_projection(pooled_features)
        
        return output


class ContrastiveFusion(nn.Module):
    """
    Contrastive learning-based multimodal fusion
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config.contrastive_temperature
        
        # Projection heads for contrastive learning
        self.text_projector = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size // 2)
        )
        
        self.audio_projector = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size // 2)
        )
        
        self.video_projector = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size // 2)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_hidden_size * 3, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        compute_contrastive_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # Project features for contrastive learning
        text_proj = F.normalize(self.text_projector(text_features), dim=-1)
        audio_proj = F.normalize(self.audio_projector(audio_features), dim=-1)
        video_proj = F.normalize(self.video_projector(video_features), dim=-1)
        
        # Compute contrastive losses if requested
        contrastive_losses = {}
        if compute_contrastive_loss:
            contrastive_losses['text_audio'] = self.contrastive_loss(text_proj, audio_proj)
            contrastive_losses['text_video'] = self.contrastive_loss(text_proj, video_proj)
            contrastive_losses['audio_video'] = self.contrastive_loss(audio_proj, video_proj)
        
        # Fusion
        fused_features = torch.cat([text_features, audio_features, video_features], dim=-1)
        output = self.fusion_layer(fused_features)
        
        return {
            'fused_features': output,
            'text_proj': text_proj,
            'audio_proj': audio_proj,
            'video_proj': video_proj,
            'contrastive_losses': contrastive_losses
        }
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two modalities"""
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Create positive pairs (diagonal elements)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute cross-entropy loss
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_1 + loss_2) / 2


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion with attention-based weighting
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Attention mechanism for adaptive weighting
        self.attention = nn.MultiheadAttention(
            config.fusion_hidden_size, 
            config.fusion_num_heads,
            dropout=config.fusion_dropout,
            batch_first=True
        )
        
        # Modality-specific transformations
        self.text_transform = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        self.audio_transform = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        self.video_transform = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        
        # Fusion weights predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(config.fusion_hidden_size * 3, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # Transform features
        text_transformed = self.text_transform(text_features)
        audio_transformed = self.audio_transform(audio_features)
        video_transformed = self.video_transform(video_features)
        
        # Stack features for attention
        stacked_features = torch.stack([
            text_transformed, audio_transformed, video_transformed
        ], dim=1)  # [batch_size, 3, hidden_size]
        
        # Apply self-attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Predict adaptive weights
        concat_features = torch.cat([text_features, audio_features, video_features], dim=-1)
        adaptive_weights = self.weight_predictor(concat_features)
        
        # Weighted fusion
        weighted_features = torch.sum(
            attended_features * adaptive_weights.unsqueeze(-1), dim=1
        )
        
        # Final transformation
        output = self.fusion_layer(weighted_features)
        
        return {
            'fused_features': output,
            'attention_weights': attention_weights,
            'adaptive_weights': adaptive_weights
        }


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion combining multiple fusion strategies
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multiple fusion methods
        self.early_fusion = EarlyFusion(config)
        self.mult_fusion = MultimodalTransformer(config)
        self.graph_fusion = GraphFusion(config)
        self.contrastive_fusion = ContrastiveFusion(config)
        self.adaptive_fusion = AdaptiveFusion(config)
        
        # Meta-fusion layer
        self.meta_fusion = nn.Sequential(
            nn.Linear(config.fusion_hidden_size * 5, config.fusion_hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_size * 2, config.fusion_hidden_size)
        )
        
    def forward(
        self, 
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        compute_contrastive_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # Apply different fusion methods
        early_output = self.early_fusion(text_features, audio_features, video_features)
        
        mult_output = self.mult_fusion(text_features, audio_features, video_features)
        mult_features = mult_output['fused_features']
        
        graph_output = self.graph_fusion(text_features, audio_features, video_features)
        
        contrastive_output = self.contrastive_fusion(
            text_features, audio_features, video_features, compute_contrastive_loss
        )
        contrastive_features = contrastive_output['fused_features']
        
        adaptive_output = self.adaptive_fusion(text_features, audio_features, video_features)
        adaptive_features = adaptive_output['fused_features']
        
        # Meta-fusion
        all_features = torch.cat([
            early_output, mult_features, graph_output, 
            contrastive_features, adaptive_features
        ], dim=-1)
        
        final_output = self.meta_fusion(all_features)
        
        return {
            'fused_features': final_output,
            'early_features': early_output,
            'mult_features': mult_features,
            'graph_features': graph_output,
            'contrastive_features': contrastive_features,
            'adaptive_features': adaptive_features,
            'contrastive_losses': contrastive_output.get('contrastive_losses', {}),
            'attention_weights': adaptive_output.get('attention_weights'),
            'adaptive_weights': adaptive_output.get('adaptive_weights')
        }
