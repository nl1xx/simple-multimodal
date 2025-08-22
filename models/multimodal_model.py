import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from models.encoders import TextEncoder, AudioEncoder, VideoEncoder, ModalityDropout
from models.fusion_layers import (
    EarlyFusion, LateFusion, MultimodalTransformer, 
    GraphFusion, ContrastiveFusion, AdaptiveFusion, HierarchicalFusion
)


class MultimodalEmotionModel(nn.Module):
    """
    Complete multimodal emotion recognition model
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoders
        self.text_encoder = TextEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)
        
        # Modality dropout for robustness
        self.modality_dropout = ModalityDropout(dropout_rate=0.1)
        
        # Fusion strategies
        self.fusion_type = getattr(config, 'fusion_type', 'hierarchical')
        
        if self.fusion_type == 'early':
            self.fusion_layer = EarlyFusion(config)
        elif self.fusion_type == 'late':
            self.fusion_layer = LateFusion(config)
        elif self.fusion_type == 'mult':
            self.fusion_layer = MultimodalTransformer(config)
        elif self.fusion_type == 'graph':
            self.fusion_layer = GraphFusion(config)
        elif self.fusion_type == 'contrastive':
            self.fusion_layer = ContrastiveFusion(config)
        elif self.fusion_type == 'adaptive':
            self.fusion_layer = AdaptiveFusion(config)
        elif self.fusion_type == 'hierarchical':
            self.fusion_layer = HierarchicalFusion(config)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Emotion classifier
        if self.fusion_type == 'late':
            # Late fusion handles classification internally
            self.classifier = None
        else:
            self.classifier = EmotionClassifier(config)
        
        # Auxiliary losses for multi-task learning
        self.valence_regressor = nn.Linear(config.fusion_hidden_size, 1)
        self.arousal_regressor = nn.Linear(config.fusion_hidden_size, 1)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        
    def forward(
        self, 
        text_input: Dict[str, torch.Tensor],
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        use_adapter: bool = False,
        use_prompt: bool = False,
        compute_contrastive_loss: bool = False,
        missing_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = text_input['input_ids'].size(0)
        device = text_input['input_ids'].device
        
        # Handle missing modalities
        if missing_modalities:
            if 'text' in missing_modalities:
                text_input = {
                    'input_ids': torch.zeros_like(text_input['input_ids']),
                    'attention_mask': torch.zeros_like(text_input['attention_mask'])
                }
            if 'audio' in missing_modalities:
                audio_input = torch.zeros_like(audio_input)
            if 'video' in missing_modalities:
                video_input = torch.zeros_like(video_input)
        
        # Encode modalities
        text_output = self.text_encoder(
            text_input['input_ids'], 
            text_input['attention_mask'],
            use_adapter=use_adapter,
            use_prompt=use_prompt
        )
        text_features = text_output['features']
        
        audio_output = self.audio_encoder(audio_input, use_adapter=use_adapter)
        audio_features = audio_output['features']
        
        video_output = self.video_encoder(video_input, use_adapter=use_adapter)
        video_features = video_output['features']
        
        # Apply modality dropout during training
        if self.training:
            text_features, audio_features, video_features = self.modality_dropout(
                text_features, audio_features, video_features, training=True
            )
        
        # Fusion
        if self.fusion_type == 'late':
            fusion_output = self.fusion_layer(text_features, audio_features, video_features)
            emotion_logits = fusion_output['fused_logits']
            
            # Additional outputs for late fusion
            individual_logits = {
                'text': fusion_output['text_logits'],
                'audio': fusion_output['audio_logits'],
                'video': fusion_output['video_logits']
            }
            fusion_weights = fusion_output['fusion_weights']
            
        elif self.fusion_type == 'contrastive':
            fusion_output = self.fusion_layer(
                text_features, audio_features, video_features,
                compute_contrastive_loss=compute_contrastive_loss
            )
            fused_features = fusion_output['fused_features']
            emotion_logits = self.classifier(fused_features)
            
        elif self.fusion_type == 'hierarchical':
            fusion_output = self.fusion_layer(
                text_features, audio_features, video_features,
                compute_contrastive_loss=compute_contrastive_loss
            )
            fused_features = fusion_output['fused_features']
            emotion_logits = self.classifier(fused_features)
            
        else:
            fusion_output = self.fusion_layer(text_features, audio_features, video_features)
            if isinstance(fusion_output, dict):
                fused_features = fusion_output['fused_features']
            else:
                fused_features = fusion_output
            emotion_logits = self.classifier(fused_features)
        
        # Auxiliary predictions
        if self.fusion_type != 'late':
            valence_pred = self.valence_regressor(fused_features)
            arousal_pred = self.arousal_regressor(fused_features)
            uncertainty_logits = self.uncertainty_head(fused_features)
        else:
            # For late fusion, use averaged features
            avg_features = (text_features + audio_features + video_features) / 3
            valence_pred = self.valence_regressor(avg_features)
            arousal_pred = self.arousal_regressor(avg_features)
            uncertainty_logits = self.uncertainty_head(avg_features)
        
        # Prepare output
        output = {
            'emotion_logits': emotion_logits,
            'emotion_probs': F.softmax(emotion_logits, dim=-1),
            'valence': valence_pred,
            'arousal': arousal_pred,
            'uncertainty': F.softmax(uncertainty_logits, dim=-1),
            'text_features': text_features,
            'audio_features': audio_features,
            'video_features': video_features
        }
        
        # Add fusion-specific outputs
        if self.fusion_type == 'late':
            output.update({
                'individual_logits': individual_logits,
                'fusion_weights': fusion_weights
            })
        
        if isinstance(fusion_output, dict):
            # Add attention weights and other fusion outputs
            for key, value in fusion_output.items():
                if key not in ['fused_features']:
                    output[key] = value
        
        return output


class EmotionClassifier(nn.Module):
    """
    Emotion classification head with multiple prediction strategies
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_size // 2, config.num_emotions)
        )
        
        # Hierarchical emotion classification
        # First level: positive vs negative vs neutral
        self.sentiment_classifier = nn.Linear(config.fusion_hidden_size, 3)
        
        # Second level: specific emotions within each sentiment
        self.positive_classifier = nn.Linear(config.fusion_hidden_size, 2)  # happy, surprise
        self.negative_classifier = nn.Linear(config.fusion_hidden_size, 4)  # sad, angry, fear, disgust
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Main classification
        main_logits = self.classifier(features)
        
        # Hierarchical classification (can be used for auxiliary loss)
        sentiment_logits = self.sentiment_classifier(features)
        positive_logits = self.positive_classifier(features)
        negative_logits = self.negative_classifier(features)
        
        return main_logits


class KnowledgeDistillationModel(nn.Module):
    """
    Knowledge distillation wrapper for model compression
    """
    def __init__(self, teacher_model: MultimodalEmotionModel, student_config):
        super().__init__()
        self.teacher = teacher_model
        self.student = MultimodalEmotionModel(student_config)
        self.temperature = student_config.distill_temperature
        self.alpha = student_config.distill_alpha
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        # Student forward pass
        student_output = self.student(*args, **kwargs)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher(*args, **kwargs)
        
        # Compute distillation loss
        student_logits = student_output['emotion_logits']
        teacher_logits = teacher_output['emotion_logits']
        
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            soft_student, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Add distillation loss to student output
        student_output['distillation_loss'] = distillation_loss
        student_output['teacher_logits'] = teacher_logits
        
        return student_output


class FewShotModel(nn.Module):
    """
    Few-shot learning wrapper with prompt tuning and adapters
    """
    def __init__(self, base_model: MultimodalEmotionModel, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Meta-learning components
        self.support_encoder = nn.LSTM(
            config.fusion_hidden_size, 
            config.fusion_hidden_size // 2,
            batch_first=True, 
            bidirectional=True
        )
        
        self.query_encoder = nn.LSTM(
            config.fusion_hidden_size,
            config.fusion_hidden_size // 2, 
            batch_first=True,
            bidirectional=True
        )
        
        # Prototype network
        self.prototype_network = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        )
        
    def forward(
        self, 
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        n_way: int,
        n_shot: int
    ) -> Dict[str, torch.Tensor]:
        
        # Extract features for support set
        support_features = self._extract_features(support_data)
        
        # Extract features for query set  
        query_features = self._extract_features(query_data)
        
        # Compute prototypes for each class
        prototypes = self._compute_prototypes(support_features, n_way, n_shot)
        
        # Compute distances and predictions
        distances = self._compute_distances(query_features, prototypes)
        predictions = F.softmax(-distances, dim=-1)
        
        return {
            'predictions': predictions,
            'distances': distances,
            'prototypes': prototypes,
            'support_features': support_features,
            'query_features': query_features
        }
    
    def _extract_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features using base model"""
        output = self.base_model(
            text_input=data['text'],
            audio_input=data['audio'], 
            video_input=data['video'],
            use_adapter=True,
            use_prompt=True
        )
        return output['text_features'] + output['audio_features'] + output['video_features']
    
    def _compute_prototypes(
        self, 
        support_features: torch.Tensor, 
        n_way: int, 
        n_shot: int
    ) -> torch.Tensor:
        """Compute class prototypes from support set"""
        # Reshape support features: [n_way * n_shot, feature_dim] -> [n_way, n_shot, feature_dim]
        support_features = support_features.view(n_way, n_shot, -1)
        
        # Compute prototypes as mean of support examples
        prototypes = torch.mean(support_features, dim=1)  # [n_way, feature_dim]
        
        # Apply prototype network
        prototypes = self.prototype_network(prototypes)
        
        return prototypes
    
    def _compute_distances(
        self, 
        query_features: torch.Tensor, 
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances between query features and prototypes"""
        # Euclidean distance
        distances = torch.cdist(query_features, prototypes, p=2)
        return distances


class RobustMultimodalModel(nn.Module):
    """
    Robust model handling missing modalities
    """
    def __init__(self, config):
        super().__init__()
        self.base_model = MultimodalEmotionModel(config)
        self.config = config
        
        # Modality-specific backup classifiers
        self.text_only_classifier = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        self.audio_only_classifier = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        self.video_only_classifier = nn.Linear(config.fusion_hidden_size, config.num_emotions)
        
        # Modality availability predictor
        self.modality_predictor = nn.Sequential(
            nn.Linear(config.fusion_hidden_size * 3, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_size, 3),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        text_input: Dict[str, torch.Tensor],
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        available_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Get base model output
        output = self.base_model(
            text_input=text_input,
            audio_input=audio_input,
            video_input=video_input
        )
        
        text_features = output['text_features']
        audio_features = output['audio_features'] 
        video_features = output['video_features']
        
        # Predict modality availability
        concat_features = torch.cat([text_features, audio_features, video_features], dim=-1)
        modality_availability = self.modality_predictor(concat_features)
        
        # Individual modality predictions
        text_pred = self.text_only_classifier(text_features)
        audio_pred = self.audio_only_classifier(audio_features)
        video_pred = self.video_only_classifier(video_features)
        
        # Adaptive fusion based on availability
        if available_modalities is None:
            # Use predicted availability
            weights = modality_availability
        else:
            # Use ground truth availability
            weights = torch.zeros_like(modality_availability)
            if 'text' in available_modalities:
                weights[:, 0] = 1.0
            if 'audio' in available_modalities:
                weights[:, 1] = 1.0
            if 'video' in available_modalities:
                weights[:, 2] = 1.0
        
        # Normalize weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        
        # Weighted prediction
        robust_prediction = (
            weights[:, 0:1] * text_pred +
            weights[:, 1:2] * audio_pred + 
            weights[:, 2:3] * video_pred
        )
        
        output.update({
            'robust_prediction': robust_prediction,
            'modality_availability': modality_availability,
            'individual_predictions': {
                'text': text_pred,
                'audio': audio_pred,
                'video': video_pred
            },
            'modality_weights': weights
        })
        
        return output


def create_model(config, model_type: str = 'standard') -> nn.Module:
    """
    Factory function to create different model variants
    """
    if model_type == 'standard':
        return MultimodalEmotionModel(config)
    elif model_type == 'few_shot':
        base_model = MultimodalEmotionModel(config)
        return FewShotModel(base_model, config)
    elif model_type == 'robust':
        return RobustMultimodalModel(config)
    elif model_type == 'distillation':
        # Requires pre-trained teacher model
        teacher_model = MultimodalEmotionModel(config)
        return KnowledgeDistillationModel(teacher_model, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(checkpoint_path: str, config) -> MultimodalEmotionModel:
    """
    Load pre-trained model from checkpoint
    """
    model = MultimodalEmotionModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model
