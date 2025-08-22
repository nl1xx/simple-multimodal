#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script provides detailed evaluation of trained multimodal emotion recognition models
including performance metrics, visualizations, and analysis reports.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Import our modules
from config import ModelConfig
from data.dataset_loaders import get_dataset, create_dataloader
from models.multimodal_model import MultimodalEmotionModel, load_pretrained_model

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = ModelConfig(**config_dict['model_config'])
        else:
            self.config = ModelConfig()
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_pretrained_model(model_path, self.config)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def evaluate_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate model on a dataset"""
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_features = []
        individual_modality_preds = {'text': [], 'audio': [], 'video': []}
        
        total_samples = 0
        correct_predictions = 0
        
        print("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    text_input=batch['text'],
                    audio_input=batch['audio'],
                    video_input=batch['video']
                )
                
                # Get predictions
                logits = outputs['emotion_logits']
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['emotion'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Collect features for visualization
                if 'text_features' in outputs:
                    features = (
                        outputs['text_features'] + 
                        outputs['audio_features'] + 
                        outputs['video_features']
                    ) / 3
                    all_features.extend(features.cpu().numpy())
                
                # Individual modality predictions (if available)
                if 'individual_logits' in outputs:
                    for modality, logits in outputs['individual_logits'].items():
                        preds = torch.argmax(logits, dim=-1)
                        individual_modality_preds[modality].extend(preds.cpu().numpy())
                
                # Update counters
                total_samples += batch['emotion'].size(0)
                correct_predictions += (predictions == batch['emotion']).sum().item()
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_features = np.array(all_features) if all_features else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Individual modality metrics
        individual_metrics = {}
        for modality, preds in individual_modality_preds.items():
            if preds:
                preds = np.array(preds)
                individual_metrics[modality] = {
                    'accuracy': accuracy_score(all_targets, preds),
                    'f1_macro': f1_score(all_targets, preds, average='macro'),
                    'f1_weighted': f1_score(all_targets, preds, average='weighted')
                }
        
        return {
            'metrics': metrics,
            'individual_metrics': individual_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'features': all_features
        }
    
    def _calculate_metrics(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray, 
        probabilities: np.ndarray
    ) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        f1_macro = f1_score(targets, predictions, average='macro')
        f1_weighted = f1_score(targets, predictions, average='weighted')
        f1_micro = f1_score(targets, predictions, average='micro')
        
        precision_macro = precision_score(targets, predictions, average='macro')
        precision_weighted = precision_score(targets, predictions, average='weighted')
        
        recall_macro = recall_score(targets, predictions, average='macro')
        recall_weighted = recall_score(targets, predictions, average='weighted')
        
        # Per-class metrics
        per_class_f1 = f1_score(targets, predictions, average=None)
        per_class_precision = precision_score(targets, predictions, average=None)
        per_class_recall = recall_score(targets, predictions, average=None)
        
        # Classification report
        class_report = classification_report(
            targets, predictions, 
            target_names=self.config.emotion_labels,
            output_dict=True
        )
        
        # ROC AUC (for multiclass)
        try:
            roc_auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
        except:
            roc_auc = None
        
        # Confidence analysis
        max_probs = np.max(probabilities, axis=1)
        correct_mask = (predictions == targets)
        
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'mean_confidence_correct': np.mean(max_probs[correct_mask]),
            'mean_confidence_incorrect': np.mean(max_probs[~correct_mask]) if np.any(~correct_mask) else 0,
            'confidence_std': np.std(max_probs)
        }
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'roc_auc': roc_auc,
            'per_class_f1': per_class_f1.tolist(),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'classification_report': class_report,
            'confidence_stats': confidence_stats
        }
    
    def create_visualizations(self, results: Dict, save_dir: str):
        """Create comprehensive visualizations"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        targets = results['targets']
        predictions = results['predictions']
        probabilities = results['probabilities']
        features = results['features']
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(targets, predictions, save_path)
        
        # 2. Per-class Performance
        self._plot_per_class_performance(results['metrics'], save_path)
        
        # 3. Confidence Distribution
        self._plot_confidence_distribution(probabilities, targets, predictions, save_path)
        
        # 4. ROC Curves
        self._plot_roc_curves(targets, probabilities, save_path)
        
        # 5. Feature Visualization (t-SNE)
        if features is not None:
            self._plot_feature_tsne(features, targets, save_path)
        
        # 6. Error Analysis
        self._plot_error_analysis(targets, predictions, probabilities, save_path)
        
        # 7. Individual Modality Comparison
        if results['individual_metrics']:
            self._plot_modality_comparison(results['individual_metrics'], save_path)
        
        print(f"Visualizations saved to: {save_path}")
    
    def _plot_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray, save_path: Path):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(targets, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.config.emotion_labels,
            yticklabels=self.config.emotion_labels,
            ax=ax1
        )
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=self.config.emotion_labels,
            yticklabels=self.config.emotion_labels,
            ax=ax2
        )
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_performance(self, metrics: Dict, save_path: Path):
        """Plot per-class performance metrics"""
        
        emotions = self.config.emotion_labels
        f1_scores = metrics['per_class_f1']
        precision_scores = metrics['per_class_precision']
        recall_scores = metrics['per_class_recall']
        
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, f1_scores, width, label='F1-Score', alpha=0.8)
        bars2 = ax.bar(x, precision_scores, width, label='Precision', alpha=0.8)
        bars3 = ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(
        self, 
        probabilities: np.ndarray, 
        targets: np.ndarray, 
        predictions: np.ndarray, 
        save_path: Path
    ):
        """Plot confidence distribution analysis"""
        
        max_probs = np.max(probabilities, axis=1)
        correct_mask = (predictions == targets)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        ax1.hist(max_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Confidence by correctness
        ax2.hist(max_probs[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
        ax2.hist(max_probs[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution by Correctness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
            if np.any(mask):
                acc = np.mean(correct_mask[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        ax3.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
        ax3.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Reliability Diagram')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Per-emotion confidence
        emotion_confidences = []
        for emotion_id in range(len(self.config.emotion_labels)):
            mask = targets == emotion_id
            if np.any(mask):
                emotion_confidences.append(max_probs[mask])
            else:
                emotion_confidences.append([])
        
        ax4.boxplot(emotion_confidences, labels=self.config.emotion_labels)
        ax4.set_xlabel('Emotions')
        ax4.set_ylabel('Confidence')
        ax4.set_title('Confidence Distribution by Emotion')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, targets: np.ndarray, probabilities: np.ndarray, save_path: Path):
        """Plot ROC curves for each class"""
        
        n_classes = len(self.config.emotion_labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, (emotion, color) in enumerate(zip(self.config.emotion_labels, colors)):
            # Binary classification for this emotion
            binary_targets = (targets == i).astype(int)
            emotion_probs = probabilities[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(binary_targets, emotion_probs)
                auc = roc_auc_score(binary_targets, emotion_probs)
                
                ax.plot(fpr, tpr, color=color, linewidth=2, 
                       label=f'{emotion} (AUC = {auc:.3f})')
            except:
                continue
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves for Each Emotion Class')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_tsne(self, features: np.ndarray, targets: np.ndarray, save_path: Path):
        """Plot t-SNE visualization of learned features"""
        
        print("Computing t-SNE visualization...")
        
        # Sample subset for t-SNE if too many samples
        if len(features) > 5000:
            indices = np.random.choice(len(features), 5000, replace=False)
            features_subset = features[indices]
            targets_subset = targets[indices]
        else:
            features_subset = features
            targets_subset = targets
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_subset)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.config.emotion_labels)))
        
        for i, (emotion, color) in enumerate(zip(self.config.emotion_labels, colors)):
            mask = targets_subset == i
            if np.any(mask):
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[color], label=emotion, alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('t-SNE Visualization of Learned Features')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'feature_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray, 
        probabilities: np.ndarray, 
        save_path: Path
    ):
        """Plot error analysis"""
        
        # Find misclassified samples
        incorrect_mask = predictions != targets
        incorrect_targets = targets[incorrect_mask]
        incorrect_predictions = predictions[incorrect_mask]
        incorrect_probs = probabilities[incorrect_mask]
        
        if len(incorrect_targets) == 0:
            print("No misclassifications found!")
            return
        
        # Error confusion matrix
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Misclassification patterns
        error_cm = confusion_matrix(incorrect_targets, incorrect_predictions)
        sns.heatmap(
            error_cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=self.config.emotion_labels,
            yticklabels=self.config.emotion_labels,
            ax=ax1
        )
        ax1.set_title('Misclassification Patterns')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Error rate by true class
        error_rates = []
        for i in range(len(self.config.emotion_labels)):
            class_mask = targets == i
            if np.any(class_mask):
                error_rate = np.mean(predictions[class_mask] != targets[class_mask])
                error_rates.append(error_rate)
            else:
                error_rates.append(0)
        
        ax2.bar(self.config.emotion_labels, error_rates, color='red', alpha=0.7)
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Rate by True Class')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Confidence of misclassified samples
        incorrect_max_probs = np.max(incorrect_probs, axis=1)
        ax3.hist(incorrect_max_probs, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Confidence Distribution of Misclassified Samples')
        ax3.grid(True, alpha=0.3)
        
        # Most confused pairs
        confusion_pairs = {}
        for true_label, pred_label in zip(incorrect_targets, incorrect_predictions):
            pair = (true_label, pred_label)
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Get top 10 most confused pairs
        top_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_pairs:
            pair_labels = [f"{self.config.emotion_labels[pair[0]]} → {self.config.emotion_labels[pair[1]]}" 
                          for pair, count in top_pairs]
            pair_counts = [count for pair, count in top_pairs]
            
            ax4.barh(range(len(pair_labels)), pair_counts, color='orange', alpha=0.7)
            ax4.set_yticks(range(len(pair_labels)))
            ax4.set_yticklabels(pair_labels)
            ax4.set_xlabel('Number of Misclassifications')
            ax4.set_title('Most Common Misclassification Pairs')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_modality_comparison(self, individual_metrics: Dict, save_path: Path):
        """Plot comparison of individual modality performance"""
        
        modalities = list(individual_metrics.keys())
        metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(modalities))
        width = 0.25
        
        colors = ['blue', 'green', 'red']
        
        for i, metric in enumerate(metrics):
            values = [individual_metrics[mod][metric] for mod in modalities]
            bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
                         color=colors[i], alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Modalities')
        ax.set_ylabel('Score')
        ax.set_title('Individual Modality Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels([mod.title() for mod in modalities])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'modality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict, save_path: str):
        """Generate comprehensive evaluation report"""
        
        report_path = Path(save_path) / 'evaluation_report.html'
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Emotion Recognition - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .poor {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 Multimodal Emotion Recognition Evaluation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Overall Performance</h2>
                <div class="metric">
                    <strong>Accuracy:</strong> {results['metrics']['accuracy']:.4f}
                    <span class="{'good' if results['metrics']['accuracy'] > 0.8 else 'warning' if results['metrics']['accuracy'] > 0.6 else 'poor'}">
                        ({'Excellent' if results['metrics']['accuracy'] > 0.8 else 'Good' if results['metrics']['accuracy'] > 0.6 else 'Needs Improvement'})
                    </span>
                </div>
                <div class="metric">
                    <strong>F1-Score (Macro):</strong> {results['metrics']['f1_macro']:.4f}
                </div>
                <div class="metric">
                    <strong>F1-Score (Weighted):</strong> {results['metrics']['f1_weighted']:.4f}
                </div>
                <div class="metric">
                    <strong>Precision (Macro):</strong> {results['metrics']['precision_macro']:.4f}
                </div>
                <div class="metric">
                    <strong>Recall (Macro):</strong> {results['metrics']['recall_macro']:.4f}
                </div>
                {f'<div class="metric"><strong>ROC AUC:</strong> {results["metrics"]["roc_auc"]:.4f}</div>' if results['metrics']['roc_auc'] else ''}
            </div>
            
            <div class="section">
                <h2>🎯 Per-Class Performance</h2>
                <table class="table">
                    <tr>
                        <th>Emotion</th>
                        <th>F1-Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Support</th>
                    </tr>
        """
        
        # Add per-class metrics
        class_report = results['metrics']['classification_report']
        for emotion in self.config.emotion_labels:
            if emotion in class_report:
                metrics = class_report[emotion]
                html_content += f"""
                    <tr>
                        <td>{emotion.title()}</td>
                        <td>{metrics['f1-score']:.3f}</td>
                        <td>{metrics['precision']:.3f}</td>
                        <td>{metrics['recall']:.3f}</td>
                        <td>{int(metrics['support'])}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        """
        
        # Add individual modality performance if available
        if results['individual_metrics']:
            html_content += """
            <div class="section">
                <h2>🔍 Individual Modality Performance</h2>
                <table class="table">
                    <tr>
                        <th>Modality</th>
                        <th>Accuracy</th>
                        <th>F1-Score (Macro)</th>
                        <th>F1-Score (Weighted)</th>
                    </tr>
            """
            
            for modality, metrics in results['individual_metrics'].items():
                html_content += f"""
                    <tr>
                        <td>{modality.title()}</td>
                        <td>{metrics['accuracy']:.3f}</td>
                        <td>{metrics['f1_macro']:.3f}</td>
                        <td>{metrics['f1_weighted']:.3f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add confidence analysis
        confidence_stats = results['metrics']['confidence_stats']
        html_content += f"""
            <div class="section">
                <h2>🎲 Confidence Analysis</h2>
                <div class="metric">
                    <strong>Mean Confidence:</strong> {confidence_stats['mean_confidence']:.3f}
                </div>
                <div class="metric">
                    <strong>Mean Confidence (Correct Predictions):</strong> {confidence_stats['mean_confidence_correct']:.3f}
                </div>
                <div class="metric">
                    <strong>Mean Confidence (Incorrect Predictions):</strong> {confidence_stats['mean_confidence_incorrect']:.3f}
                </div>
                <div class="metric">
                    <strong>Confidence Standard Deviation:</strong> {confidence_stats['confidence_std']:.3f}
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Visualizations</h2>
                <p>The following visualizations have been generated:</p>
                <ul>
                    <li>Confusion Matrix</li>
                    <li>Per-Class Performance</li>
                    <li>Confidence Analysis</li>
                    <li>ROC Curves</li>
                    <li>Feature t-SNE Visualization</li>
                    <li>Error Analysis</li>
                    {'<li>Modality Comparison</li>' if results['individual_metrics'] else ''}
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Evaluation report saved to: {report_path}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
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

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Emotion Recognition Model")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to model configuration')
    parser.add_argument('--data_path', type=str, default='./data', help='Data directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--dataset', type=str, default='cmu_mosei', 
                       choices=['cmu_mosei', 'meld', 'iemocap', 'multimodal'],
                       help='Dataset to evaluate on')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset ({args.split} split)...")
    dataset = get_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=args.split,
        config=evaluator.config,
        augment=False
    )
    
    data_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Run evaluation
    results = evaluator.evaluate_dataset(data_loader)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1-Score (Macro): {results['metrics']['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
    print(f"Precision (Macro): {results['metrics']['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['metrics']['recall_macro']:.4f}")
    
    if results['metrics']['roc_auc']:
        print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    
    # Individual modality results
    if results['individual_metrics']:
        print("\nIndividual Modality Performance:")
        for modality, metrics in results['individual_metrics'].items():
            print(f"  {modality.title()}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    evaluator.create_visualizations(results, args.output_dir)
    
    # Generate report
    evaluator.generate_report(results, args.output_dir)
    
    # Save detailed results
    results_path = output_path / 'detailed_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'metrics': results['metrics'],
        'individual_metrics': results['individual_metrics'],
        'predictions': results['predictions'].tolist(),
        'targets': results['targets'].tolist(),
        'probabilities': results['probabilities'].tolist()
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"All evaluation outputs saved to: {output_path}")

if __name__ == "__main__":
    main()
