import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """
    Model configuration parameters
    """
    # Text encoder
    text_model_name: str = "microsoft/deberta-v3-base"
    text_hidden_size: int = 768
    text_max_length: int = 512
    
    # Audio encoder
    audio_model_name: str = "facebook/wav2vec2-base-960h"
    audio_hidden_size: int = 768
    audio_sample_rate: int = 16000
    audio_max_length: int = 160000  # 10 seconds at 16kHz
    
    # Video encoder
    video_model_name: str = "google/vit-base-patch16-224"
    video_hidden_size: int = 768
    video_frame_size: Tuple[int, int] = (224, 224)
    video_max_frames: int = 30
    
    # Fusion parameters
    fusion_hidden_size: int = 512
    fusion_dropout: float = 0.1
    fusion_num_heads: int = 8
    fusion_num_layers: int = 4
    
    # Emotion classification
    num_emotions: int = 7  # happy, sad, angry, fear, surprise, disgust, neutral
    emotion_labels: List[str] = None
    
    # Graph fusion parameters
    graph_hidden_size: int = 256
    graph_num_layers: int = 3
    graph_dropout: float = 0.1
    
    # Contrastive learning parameters
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.5
    
    # Few-shot learning parameters
    adapter_size: int = 64
    prompt_length: int = 10
    
    # Knowledge distillation parameters
    distill_temperature: float = 4.0
    distill_alpha: float = 0.7
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # Data paths
    data_path: str = "./data"
    save_path: str = "./checkpoints"
    log_path: str = "./logs"
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.emotion_labels is None:
            self.emotion_labels = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        
        # Create directories
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)


@dataclass
class DataConfig:
    """
    Data configuration parameters
    """
    # Dataset selection
    primary_dataset: str = "sample"  # cmu_mosei, meld, iemocap...
    supplementary_datasets: List[str] = None
    
    # Data preprocessing
    normalize_audio: bool = True
    augment_data: bool = True
    balance_classes: bool = True
    
    # Cross-validation
    k_folds: int = 5
    test_split: float = 0.2
    val_split: float = 0.1
    
    # Data loading
    num_workers: int = 0
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.supplementary_datasets is None:
            self.supplementary_datasets = ["meld"]


@dataclass
class ExperimentConfig:
    """
    Experiment configuration for research
    """
    # Ablation studies
    enable_early_fusion: bool = True
    enable_late_fusion: bool = True
    enable_mult_fusion: bool = True
    enable_graph_fusion: bool = True
    enable_contrastive_learning: bool = True
    
    # Few-shot learning
    enable_prompt_tuning: bool = True
    enable_adapter_tuning: bool = True
    few_shot_samples: List[int] = None
    
    # Robustness testing
    test_missing_modalities: bool = True
    missing_modality_rates: List[float] = None
    
    # Knowledge distillation
    enable_knowledge_distillation: bool = True
    teacher_model_path: Optional[str] = None
    
    def __post_init__(self):
        if self.few_shot_samples is None:
            self.few_shot_samples = [1, 5, 10, 20, 50]
        
        if self.missing_modality_rates is None:
            self.missing_modality_rates = [0.1, 0.3, 0.5, 0.7]


# Global configuration instances
model_config = ModelConfig()
data_config = DataConfig()
experiment_config = ExperimentConfig()
