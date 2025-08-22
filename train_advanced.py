"""
Advanced Training Script for Multimodal Emotion Recognition System

- Multiple fusion strategies (Early, Late, MulT, Graph, Contrastive, Hierarchical)
- Few-shot learning with prompt tuning and adapters
- Knowledge distillation for model compression
- Robustness training with missing modalities
- Contrastive learning for cross-modal alignment
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import wandb
from typing import Dict
from config import ModelConfig, DataConfig, ExperimentConfig
from data.dataset_loaders import get_dataset, create_dataloader, FewShotDataset
from models.multimodal_model import (MultimodalEmotionModel, KnowledgeDistillationModel, FewShotModel,
                                     RobustMultimodalModel, create_model)
from training.advanced_trainer import AdvancedTrainer, FewShotTrainer, RobustnessTrainer


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(data_config: DataConfig, model_config: ModelConfig) -> Dict[str, DataLoader]:
    """
    Load and prepare datasets
    """
    print("Loading datasets...")
    
    # Load primary dataset
    train_dataset = get_dataset(
        dataset_name=data_config.primary_dataset,
        data_path=model_config.data_path,
        split='train',
        config=model_config,
        augment=data_config.augment_data
    )
    
    val_dataset = get_dataset(
        dataset_name=data_config.primary_dataset,
        data_path=model_config.data_path,
        split='val',
        config=model_config,
        augment=False
    )
    
    test_dataset = get_dataset(
        dataset_name=data_config.primary_dataset,
        data_path=model_config.data_path,
        split='test',
        config=model_config,
        augment=False
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def train_standard_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    fusion_type: str = 'hierarchical'
) -> str:
    """
    Train standard multimodal model
    """
    print(f"=== Training Standard Model with {fusion_type} fusion ===")
    
    # Update fusion type in config
    model_config.fusion_type = fusion_type
    
    # Load datasets
    dataloaders = load_datasets(data_config, model_config)
    
    # Create model
    model = create_model(model_config, model_type='standard')
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        config=model_config,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test']
    )
    
    # Train model
    training_history = trainer.train()
    
    # Save final model
    model_path = Path(model_config.save_path) / f'final_model_{fusion_type}.pth'
    trainer.save_checkpoint(str(model_path), trainer.current_epoch, {})
    
    print(f"Model saved to: {model_path}")
    return str(model_path)


def train_few_shot_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    experiment_config: ExperimentConfig
) -> Dict[str, float]:
    """
    Train and evaluate few-shot learning model
    """
    print(f"=== Few-Shot Learning Experiments ===")
    
    results = {}
    
    # Load base dataset
    train_dataset = get_dataset(
        dataset_name=data_config.primary_dataset,
        data_path=model_config.data_path,
        split='train',
        config=model_config,
        augment=False
    )
    
    val_dataset = get_dataset(
        dataset_name=data_config.primary_dataset,
        data_path=model_config.data_path,
        split='val',
        config=model_config,
        augment=False
    )
    
    # Test different few-shot scenarios
    for n_shot in experiment_config.few_shot_samples:
        print(f"Training {n_shot}-shot model...")
        
        # Create few-shot datasets
        few_shot_train = FewShotDataset(
            base_dataset=train_dataset,
            n_shot=n_shot,
            n_way=model_config.num_emotions
        )
        
        few_shot_val = FewShotDataset(
            base_dataset=val_dataset,
            n_shot=n_shot,
            n_way=model_config.num_emotions
        )
        
        # Create data loaders
        support_loader = create_dataloader(few_shot_train, batch_size=n_shot * model_config.num_emotions, shuffle=True)
        query_loader = create_dataloader(few_shot_val, batch_size=16, shuffle=False)
        
        # Create few-shot model
        base_model = create_model(model_config, model_type='standard')
        few_shot_model = FewShotModel(base_model, model_config)
        
        # Create trainer
        trainer = FewShotTrainer(
            model=few_shot_model,
            config=model_config,
            support_loader=support_loader,
            query_loader=query_loader
        )
        
        # Train for a few episodes
        total_loss = 0
        num_episodes = 100
        
        for episode in range(num_episodes):
            loss = trainer.train_few_shot_episode(
                n_way=model_config.num_emotions,
                n_shot=n_shot
            )
            total_loss += loss
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_episodes
        results[f'{n_shot}_shot'] = avg_loss
        
        print(f"{n_shot}-shot average loss: {avg_loss:.4f}")
    
    return results


def train_knowledge_distillation(
    model_config: ModelConfig,
    data_config: DataConfig,
    teacher_model_path: str
) -> str:
    """
    Train student model with knowledge distillation
    """
    print(f"=== Knowledge Distillation Training ===")
    
    # Load datasets
    dataloaders = load_datasets(data_config, model_config)
    
    # Load teacher model
    teacher_model = create_model(model_config, model_type='standard')
    teacher_checkpoint = torch.load(teacher_model_path, map_location='cpu')
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    
    # Create student config (smaller model)
    student_config = ModelConfig()
    student_config.fusion_hidden_size = model_config.fusion_hidden_size // 2
    student_config.fusion_num_heads = model_config.fusion_num_heads // 2
    student_config.fusion_num_layers = model_config.fusion_num_layers // 2
    
    # Create distillation model
    distillation_model = KnowledgeDistillationModel(teacher_model, student_config)
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=distillation_model,
        config=student_config,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test']
    )
    
    # Train model
    training_history = trainer.train()
    
    # Save student model
    student_path = Path(model_config.save_path) / 'distilled_student_model.pth'
    torch.save(distillation_model.student.state_dict(), student_path)
    
    print(f"Distilled model saved to: {student_path}")
    return str(student_path)


def train_robust_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    experiment_config: ExperimentConfig
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate robust model with missing modalities
    """
    print(f"=== Robustness Training ===")
    
    # Load datasets
    dataloaders = load_datasets(data_config, model_config)
    
    # Create robust model
    robust_model = create_model(model_config, model_type='robust')
    
    # Create trainer
    trainer = RobustnessTrainer(
        model=robust_model,
        config=model_config,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test']
    )
    
    # Train with missing modalities
    print("Training with random modality dropout...")
    for epoch in range(model_config.num_epochs // 2):  # Train for half epochs
        trainer.current_epoch = epoch
        train_metrics = trainer.train_with_missing_modalities()
        print(f"Epoch {epoch + 1}, Loss: {train_metrics['avg_loss']:.4f}")
    
    # Evaluate robustness
    print("Evaluating robustness...")
    robustness_results = trainer.evaluate_robustness()
    
    # Print results
    print("Robustness Results:")
    for scenario, metrics in robustness_results.items():
        print(f"{scenario}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")
    
    # Save robust model
    robust_path = Path(model_config.save_path) / 'robust_model.pth'
    trainer.save_checkpoint(str(robust_path), trainer.current_epoch, {})
    
    return robustness_results


def run_ablation_studies(
    model_config: ModelConfig,
    data_config: DataConfig,
    experiment_config: ExperimentConfig
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation studies on different fusion methods
    """
    print(f"=== Ablation Studies ===")
    
    fusion_methods = []
    if experiment_config.enable_early_fusion:
        fusion_methods.append('early')
    if experiment_config.enable_late_fusion:
        fusion_methods.append('late')
    if experiment_config.enable_mult_fusion:
        fusion_methods.append('mult')
    if experiment_config.enable_graph_fusion:
        fusion_methods.append('graph')
    if experiment_config.enable_contrastive_learning:
        fusion_methods.append('contrastive')
    
    results = {}
    
    for fusion_type in fusion_methods:
        print(f"Testing {fusion_type} fusion...")
        
        # Create temporary config
        temp_config = ModelConfig()
        temp_config.__dict__.update(model_config.__dict__)
        temp_config.fusion_type = fusion_type
        temp_config.num_epochs = 10  # Shorter training for ablation
        
        # Load datasets
        dataloaders = load_datasets(data_config, temp_config)
        
        # Create model
        model = create_model(temp_config, model_type='standard')
        
        # Create trainer
        trainer = AdvancedTrainer(
            model=model,
            config=temp_config,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test']
        )
        
        # Quick training
        training_history = trainer.train()
        
        # Get final metrics
        final_metrics = {
            'val_accuracy': trainer.best_val_acc,
            'val_f1': trainer.best_val_f1
        }
        
        results[fusion_type] = final_metrics
        print(f"{fusion_type} - Val Acc: {final_metrics['val_accuracy']:.3f}, Val F1: {final_metrics['val_f1']:.3f}")
    
    return results


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description="Advanced Multimodal Emotion Recognition Training")
    
    # Training mode
    parser.add_argument('--mode', type=str, default='standard', 
                       choices=['standard', 'few_shot', 'distillation', 'robust', 'ablation', 'all'],
                       help='Training mode')
    
    # Model configuration
    parser.add_argument('--fusion_type', type=str, default='hierarchical',
                       choices=['early', 'late', 'mult', 'graph', 'contrastive', 'adaptive', 'hierarchical'],
                       help='Fusion strategy')
    
    # Paths
    parser.add_argument('--data_path', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Model save directory')
    parser.add_argument('--teacher_model', type=str, help='Teacher model path for distillation')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='multimodal-emotion', help='W&B project name')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create configurations
    model_config = ModelConfig()
    model_config.data_path = args.data_path
    model_config.save_path = args.save_path
    model_config.batch_size = args.batch_size
    model_config.num_epochs = args.epochs
    model_config.learning_rate = args.learning_rate
    model_config.device = args.device
    model_config.use_wandb = args.use_wandb
    
    data_config = DataConfig()
    experiment_config = ExperimentConfig()
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                'model_config': vars(model_config),
                'data_config': vars(data_config),
                'experiment_config': vars(experiment_config),
                'mode': args.mode,
                'fusion_type': args.fusion_type
            }
        )
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Run training based on mode
    if args.mode == 'standard':
        model_path = train_standard_model(model_config, data_config, args.fusion_type)
        print(f"Training completed! Model saved to: {model_path}")
        
    elif args.mode == 'few_shot':
        results = train_few_shot_model(model_config, data_config, experiment_config)
        print(f"Few-shot learning results: {results}")
        
    elif args.mode == 'distillation':
        if not args.teacher_model:
            print("Error: Teacher model path required for distillation")
            return
        student_path = train_knowledge_distillation(model_config, data_config, args.teacher_model)
        print(f"Distillation completed! Student model saved to: {student_path}")
        
    elif args.mode == 'robust':
        results = train_robust_model(model_config, data_config, experiment_config)
        print(f"Robustness training completed! Results: {results}")
        
    elif args.mode == 'ablation':
        results = run_ablation_studies(model_config, data_config, experiment_config)
        print(f"Ablation studies completed! Results: {results}")
        
    elif args.mode == 'all':
        print("Running comprehensive experiments...")
        
        # 1. Standard training with different fusion methods
        fusion_methods = ['early', 'late', 'mult', 'graph', 'contrastive', 'hierarchical']
        for fusion_type in fusion_methods:
            try:
                model_path = train_standard_model(model_config, data_config, fusion_type)
                print(f"Completed {fusion_type} fusion training")
            except Exception as e:
                print(f"Error in {fusion_type} fusion: {e}")
        
        # 2. Few-shot learning
        try:
            few_shot_results = train_few_shot_model(model_config, data_config, experiment_config)
            print(f"Few-shot results: {few_shot_results}")
        except Exception as e:
            print(f"Error in few-shot learning: {e}")
        
        # 3. Robustness training
        try:
            robust_results = train_robust_model(model_config, data_config, experiment_config)
            print(f"Robustness results: {robust_results}")
        except Exception as e:
            print(f"Error in robustness training: {e}")
        
        # 4. Ablation studies
        try:
            ablation_results = run_ablation_studies(model_config, data_config, experiment_config)
            print(f"Ablation results: {ablation_results}")
        except Exception as e:
            print(f"Error in ablation studies: {e}")
    
    # Save final configuration
    config_save_path = Path(args.save_path) / 'final_config.json'
    with open(config_save_path, 'w') as f:
        json.dump({
            'model_config': vars(model_config),
            'data_config': vars(data_config),
            'experiment_config': vars(experiment_config)
        }, f, indent=2)
    
    print(f"Configuration saved to: {config_save_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
