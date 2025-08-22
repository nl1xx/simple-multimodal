import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Optional
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from models.multimodal_model import (
    MultimodalEmotionModel, KnowledgeDistillationModel, 
    FewShotModel, RobustMultimodalModel
)


class AdvancedTrainer:
    """
    Advanced trainer with multiple learning strategies
    """
    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        # Early stopping
        self.patience = getattr(config, 'patience', 10)
        self.patience_counter = 0
        
        # Logging
        self.use_wandb = getattr(config, 'use_wandb', False)
        if self.use_wandb:
            wandb.init(project="multimodal-emotion", config=vars(config))
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer with different learning rates for different components
        """
        # Different learning rates for pre-trained and new components
        pretrained_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['text_encoder.model', 'audio_encoder.model', 'video_encoder.vit']):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        optimizer = AdamW([
            {'params': pretrained_params, 'lr': self.config.learning_rate * 0.1},
            {'params': new_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Setup learning rate scheduler
        """
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        emotion_loss = 0.0
        contrastive_loss = 0.0
        auxiliary_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    text_input=batch['text'],
                    audio_input=batch['audio'],
                    video_input=batch['video'],
                    compute_contrastive_loss=True
                )
                
                # Main emotion classification loss
                emotion_logits = outputs['emotion_logits']
                emotion_targets = batch['emotion']
                main_loss = self.criterion(emotion_logits, emotion_targets)
                
                # Contrastive losses
                contrastive_losses = outputs.get('contrastive_losses', {})
                contrastive_loss_value = sum(contrastive_losses.values()) if contrastive_losses else 0.0
                
                # Auxiliary losses (valence/arousal regression)
                aux_loss = 0.0
                if 'valence' in outputs and hasattr(batch, 'valence'):
                    aux_loss += self.mse_loss(outputs['valence'].squeeze(), batch['valence'])
                if 'arousal' in outputs and hasattr(batch, 'arousal'):
                    aux_loss += self.mse_loss(outputs['arousal'].squeeze(), batch['arousal'])
                
                # Knowledge distillation loss
                distill_loss = 0.0
                if 'distillation_loss' in outputs:
                    distill_loss = outputs['distillation_loss']
                
                # Total loss
                total_loss_batch = (
                    main_loss + 
                    0.1 * contrastive_loss_value + 
                    0.1 * aux_loss +
                    0.5 * distill_loss
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            emotion_loss += main_loss.item()
            contrastive_loss += contrastive_loss_value if isinstance(contrastive_loss_value, float) else contrastive_loss_value.item()
            auxiliary_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Emotion': f'{main_loss.item():.4f}',
                'Contrastive': f'{contrastive_loss_value:.4f}' if isinstance(contrastive_loss_value, float) else f'{contrastive_loss_value.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate average losses
        num_batches = len(self.train_loader)
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'emotion_loss': emotion_loss / num_batches,
            'contrastive_loss': contrastive_loss / num_batches,
            'auxiliary_loss': auxiliary_loss / num_batches
        }
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    text_input=batch['text'],
                    audio_input=batch['audio'],
                    video_input=batch['video']
                )
                
                # Calculate loss
                emotion_logits = outputs['emotion_logits']
                emotion_targets = batch['emotion']
                loss = self.criterion(emotion_logits, emotion_targets)
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(emotion_logits, dim=-1)
                probabilities = F.softmax(emotion_logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(emotion_targets.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_f1_macro': f1_macro,
            'val_f1_weighted': f1_weighted
        }
        
        # Generate classification report
        class_report = classification_report(
            all_targets, all_predictions,
            target_names=self.config.emotion_labels,
            output_dict=True
        )
        
        return metrics, class_report, all_predictions, all_targets, all_probs
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics, class_report, predictions, targets, probs = self.validate()
            
            # Update tracking
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['val_loss'])
            self.val_accuracies.append(val_metrics['val_accuracy'])
            self.val_f1_scores.append(val_metrics['val_f1_macro'])
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"Val F1 (Macro): {val_metrics['val_f1_macro']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['val_loss'],
                    'val_accuracy': val_metrics['val_accuracy'],
                    'val_f1_macro': val_metrics['val_f1_macro'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save best model
            if val_metrics['val_f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['val_f1_macro']
                self.best_val_acc = val_metrics['val_accuracy']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                self.patience_counter = 0
                
                # Save confusion matrix for best model
                self.plot_confusion_matrix(targets, predictions, epoch)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)
        
        # Final evaluation
        if self.test_loader:
            test_metrics = self.evaluate_test_set()
            print(f"\nFinal Test Results:")
            print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
            print(f"Test F1 (Macro): {test_metrics['test_f1_macro']:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Evaluate on test set
        """
        if not self.test_loader:
            return {}
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    text_input=batch['text'],
                    audio_input=batch['audio'],
                    video_input=batch['video']
                )
                
                predictions = torch.argmax(outputs['emotion_logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['emotion'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'test_accuracy': accuracy,
            'test_f1_macro': f1_macro,
            'test_f1_weighted': f1_weighted
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """
        Move batch to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                device_batch[key] = value
        
        return device_batch
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = Path(self.config.save_path) / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def plot_confusion_matrix(self, targets: List[int], predictions: List[int], epoch: int):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.config.emotion_labels,
            yticklabels=self.config.emotion_labels
        )
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = Path(self.config.log_path) / f'confusion_matrix_epoch_{epoch + 1}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.use_wandb:
            wandb.log({f'confusion_matrix_epoch_{epoch + 1}': wandb.Image(str(save_path))})
    
    def plot_training_curves(self):
        """
        Plot and save training curves
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # F1 score curve
        ax3.plot(epochs, self.val_f1_scores, 'm-', label='Validation F1 (Macro)')
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate curve
        ax4.plot(epochs, [self.scheduler.get_last_lr()[0]] * len(epochs), 'c-', label='Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        save_path = Path(self.config.log_path) / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.use_wandb:
            wandb.log({'training_curves': wandb.Image(str(save_path))})


class FewShotTrainer(AdvancedTrainer):
    """
    Trainer for few-shot learning scenarios
    """
    def __init__(self, model: FewShotModel, config, support_loader: DataLoader, query_loader: DataLoader):
        self.model = model
        self.config = config
        self.support_loader = support_loader
        self.query_loader = query_loader
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Setup optimizer (only for adapters and prompt embeddings)
        self.optimizer = self._setup_few_shot_optimizer()
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss()
    
    def _setup_few_shot_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer for few-shot learning (only trainable parameters)
        """
        trainable_params = []
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['adapter', 'prompt_embeddings', 'prototype_network']):
                trainable_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return AdamW(trainable_params, lr=self.config.learning_rate)
    
    def train_few_shot_episode(self, n_way: int, n_shot: int) -> float:
        """
        Train on a single few-shot episode
        """
        self.model.train()
        
        # Sample support and query sets
        support_batch = next(iter(self.support_loader))
        query_batch = next(iter(self.query_loader))
        
        support_batch = self._move_batch_to_device(support_batch)
        query_batch = self._move_batch_to_device(query_batch)
        
        # Forward pass
        outputs = self.model(
            support_data=support_batch,
            query_data=query_batch,
            n_way=n_way,
            n_shot=n_shot
        )
        
        # Calculate loss
        predictions = outputs['predictions']
        targets = query_batch['emotion']
        loss = self.criterion(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class RobustnessTrainer(AdvancedTrainer):
    """
    Trainer for robustness testing with missing modalities
    """
    def train_with_missing_modalities(self) -> Dict[str, float]:
        """Train with random modality dropout"""
        
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc='Robustness Training'):
            batch = self._move_batch_to_device(batch)
            
            # Randomly select missing modalities
            missing_modalities = []
            if torch.rand(1) < 0.3:  # 30% chance of missing text
                missing_modalities.append('text')
            if torch.rand(1) < 0.3:  # 30% chance of missing audio
                missing_modalities.append('audio')
            if torch.rand(1) < 0.3:  # 30% chance of missing video
                missing_modalities.append('video')
            
            # Forward pass with missing modalities
            outputs = self.model(
                text_input=batch['text'],
                audio_input=batch['audio'],
                video_input=batch['video'],
                missing_modalities=missing_modalities
            )
            
            # Calculate loss
            if isinstance(self.model, RobustMultimodalModel):
                logits = outputs['robust_prediction']
            else:
                logits = outputs['emotion_logits']
            
            loss = self.criterion(logits, batch['emotion'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {'avg_loss': total_loss / len(self.train_loader)}
    
    def evaluate_robustness(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness with different missing modality scenarios
        """
        scenarios = [
            [],  # All modalities
            ['text'],  # Missing text
            ['audio'],  # Missing audio
            ['video'],  # Missing video
            ['text', 'audio'],  # Missing text and audio
            ['text', 'video'],  # Missing text and video
            ['audio', 'video'],  # Missing audio and video
        ]
        
        results = {}
        
        for missing_modalities in scenarios:
            scenario_name = 'all' if not missing_modalities else '_'.join(missing_modalities) + '_missing'
            
            self.model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = self._move_batch_to_device(batch)
                    
                    outputs = self.model(
                        text_input=batch['text'],
                        audio_input=batch['audio'],
                        video_input=batch['video'],
                        missing_modalities=missing_modalities
                    )
                    
                    if isinstance(self.model, RobustMultimodalModel):
                        logits = outputs['robust_prediction']
                    else:
                        logits = outputs['emotion_logits']
                    
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch['emotion'].cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            f1_macro = f1_score(all_targets, all_predictions, average='macro')
            
            results[scenario_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro
            }
        
        return results
