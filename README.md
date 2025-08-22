# Advanced Multimodal Emotion Recognition & Interactive Dialogue System

A state-of-the-art PyTorch implementation featuring all innovative algorithms for multimodal emotion recognition, combining **text**, **audio**, and **video** inputs with advanced fusion techniques.

## Project Overview

This system implements cutting-edge research in multimodal emotion recognition with:
- **6+ Fusion Strategies** including Graph Neural Networks and Contrastive Learning
- **Few-Shot Learning** with prompt tuning and adapters
- **Knowledge Distillation** for model compression
- **Robustness Training** for missing modalities
- **Interactive Demo** with real-time processing

## System Architecture

```
📁 Project Structure
├── 📄 config.py                    # Enhanced configuration system
├── 📄 requirements.txt             # Dependencies
├── 📄 train_advanced.py           # Main training script
├── 📄 evaluate_model.py           # Comprehensive evaluation
├── 📄 README.md                   # Basic documentation
├── 📄 README_COMPLETE.md          # Complete documentation
├── 📂 data/
│   └── 📄 dataset_loaders.py      # Advanced data loading
├── 📂 models/
│   ├── 📄 encoders.py             # Multimodal encoders
│   ├── 📄 fusion_layers.py        # All fusion strategies
│   └── 📄 multimodal_model.py     # Complete model system
├── 📂 training/
│   └── 📄 advanced_trainer.py     # Advanced training system
└── 📂 demo/
    └── 📄 gradio_demo.py          # Interactive web interface
```

## Innovative Algorithms Implemented

### **Fusion Strategies**
1. **Early Fusion**: Simple concatenation with MLP layers
2. **Late Fusion**: Weighted decision-level combination
3. **MulT (Multimodal Transformer)**: Cross-modal attention mechanisms
4. **Graph Fusion**: Graph Neural Networks modeling modality relationships
5. **Contrastive Fusion**: Cross-modal contrastive learning
6. **Adaptive Fusion**: Dynamic attention-based weighting
7. **Hierarchical Fusion**: Meta-fusion combining all methods

### **Advanced Learning Techniques**
- **Few-Shot Learning**: Prompt tuning + Adapter layers for low-resource scenarios
- **Knowledge Distillation**: Teacher-student framework for model compression
- **Robustness Training**: Graceful handling of missing modalities
- **Cross-Modal Contrastive Learning**: Unified semantic space learning
- **Parameter-Efficient Fine-tuning**: Minimal parameter updates

### **Model Architecture**
- **Text Encoder**: DeBERTa with adapter support and prompt tuning
- **Audio Encoder**: Wav2Vec2 with temporal attention mechanisms
- **Video Encoder**: Vision Transformer + LSTM for facial expressions
- **Modality Dropout**: Training robustness with random modality removal

## Supported Datasets

| Dataset | Modalities | Samples | Description |
|---------|------------|---------|-------------|
| **CMU-MOSEI** | Text + Audio + Video | 23,454 | Multimodal sentiment analysis |
| **MELD** | Text + Audio + Video | 13,000 | Multi-party dialogue emotions |
| **IEMOCAP** | Text + Audio + Motion | 12 hours | Interactive emotional dyadic motion capture |

## Quick Start

### 1. Installation
```bash
# Clone and install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Organize your data as:
data/
├── train.csv
├── val.csv  
├── test.csv
├── audio/
│   ├── sample_001.wav
│   └── ...
└── video/
    ├── sample_001.mp4
    └── ...
```

**CSV Format:**
```csv
text,audio_path,video_path,emotion
"I'm so happy today!",audio/happy_001.wav,video/happy_001.mp4,happy
"This is frustrating.",audio/angry_001.wav,video/angry_001.mp4,angry
```

### 3. Training

#### Standard Training (All Fusion Methods)
```bash
python train_advanced.py --mode standard --fusion_type hierarchical
```

#### Few-Shot Learning
```bash
python train_advanced.py --mode few_shot
```

#### Knowledge Distillation
```bash
# First train teacher model
python train_advanced.py --mode standard --fusion_type hierarchical

# Then distill to student
python train_advanced.py --mode distillation --teacher_model checkpoints/best_model.pth
```

#### Robustness Training
```bash
python train_advanced.py --mode robust
```

#### Complete Experimental Suite
```bash
python train_advanced.py --mode all
```

### 4. Evaluation
```bash
python evaluate_model.py --model_path checkpoints/best_model.pth --output_dir results/
```

### 5. Interactive Demo
```bash
python demo/gradio_demo.py --model_path checkpoints/best_model.pth --port 7860
```

## Configuration Options

### Model Configuration
```python
@dataclass
class ModelConfig:
    # Text processing
    text_model_name: str = "microsoft/deberta-v3-base"
    text_max_length: int = 512
    
    # Audio processing  
    audio_model_name: str = "facebook/wav2vec2-base-960h"
    audio_sample_rate: int = 16000
    
    # Video processing
    video_model_name: str = "google/vit-base-patch16-224"
    video_max_frames: int = 30
    
    # Fusion parameters
    fusion_type: str = "hierarchical"  # early, late, mult, graph, contrastive, adaptive, hierarchical
    fusion_hidden_size: int = 512
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
```

## Performance Metrics

The system provides comprehensive evaluation including:

- **Classification Metrics**: Accuracy, F1-Score (Macro/Weighted), Precision, Recall
- **Confidence Analysis**: Reliability diagrams, calibration curves
- **Per-Class Performance**: Individual emotion recognition rates
- **Modality Analysis**: Individual and combined modality performance
- **Robustness Testing**: Performance with missing modalities

## Visualization Features

### Training Monitoring
- Real-time loss and accuracy curves
- Confusion matrices with class-wise analysis
- Learning rate scheduling visualization
- Gradient flow monitoring

### Evaluation Visualizations
- t-SNE feature space visualization
- ROC curves for each emotion class
- Confidence distribution analysis
- Error pattern analysis
- Modality contribution comparison

## Advanced Features

### Few-Shot Learning
```python
# Create few-shot dataset
few_shot_dataset = FewShotDataset(
    base_dataset=train_dataset,
    n_shot=5,  # 5 examples per class
    n_way=7    # 7 emotion classes
)

# Train few-shot model
few_shot_model = FewShotModel(base_model, config)
```

### Knowledge Distillation
```python
# Create teacher-student setup
teacher_model = MultimodalEmotionModel(config)
distillation_model = KnowledgeDistillationModel(teacher_model, student_config)
```

### Robustness Testing
```python
# Test with missing modalities
robustness_results = trainer.evaluate_robustness()
# Returns performance for all missing modality scenarios
```

## Emotion Classes

| Emotion | Description | System Response Strategy |
|---------|-------------|-------------------------|
| **Happy** | Joy, excitement, satisfaction | Share positivity, encourage continuation |
| **Sad** | Sorrow, disappointment, grief | Provide comfort, empathetic listening |
| **Angry** | Frustration, irritation, rage | Calm guidance, emotion regulation |
| **Fear** | Anxiety, worry, apprehension | Reassurance, safety provision |
| **Surprise** | Amazement, shock, wonder | Share excitement, help process |
| **Disgust** | Revulsion, distaste, aversion | Understanding, attention redirection |
| **Neutral** | Calm, balanced, no strong emotion | Friendly engagement, topic exploration |

## Interactive Demo Features

### Real-Time Processing
- **Webcam Integration**: Live facial expression analysis
- **Microphone Support**: Real-time speech emotion recognition
- **Text Input**: Natural language emotion understanding

### Intelligent Responses
- **Emotion-Aware Dialogue**: Context-appropriate responses
- **Activity Suggestions**: Personalized recommendations based on detected emotions
- **Confidence Visualization**: Real-time confidence and uncertainty display

### Visualization Dashboard
- **Emotion Distribution**: Real-time probability charts
- **Valence-Arousal Space**: 2D emotion mapping
- **Individual Modality Results**: Separate text/audio/video analysis
- **Conversation History**: Track emotional patterns over time

## Research Applications

### Ablation Studies
```bash
# Compare all fusion methods
python train_advanced.py --mode ablation
```

### Cross-Dataset Evaluation
```bash
# Train on CMU-MOSEI, test on MELD
python evaluate_model.py --model_path cmu_mosei_model.pth --dataset meld
```

### Missing Modality Analysis
```bash
# Systematic robustness evaluation
python train_advanced.py --mode robust
```

