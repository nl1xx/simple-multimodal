import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import cv2
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List
from pathlib import Path


class MultimodalDataset(Dataset):
    """
    Base multimodal dataset class
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        config: 'ModelConfig' = None,
        transform_audio=None,
        transform_video=None,
        augment: bool = False
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.config = config
        self.transform_audio = transform_audio
        self.transform_video = transform_video
        self.augment = augment
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        
        # Load data
        self.data = self._load_data()
        
        # Create emotion label mapping
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(config.emotion_labels)}
        self.id_to_emotion = {i: emotion for emotion, i in self.emotion_to_id.items()}
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load dataset based on split
        """
        csv_path = self.data_path / f"{self.split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        return pd.read_csv(csv_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Process text
        text_features = self._process_text(row['text'])
        
        # Process audio
        audio_features = self._process_audio(row['audio_path'])
        
        # Process video
        video_features = self._process_video(row['video_path'])
        
        # Process emotion label
        emotion_label = self.emotion_to_id[row['emotion']]
        
        return {
            'text': text_features,
            'audio': audio_features,
            'video': video_features,
            'emotion': torch.tensor(emotion_label, dtype=torch.long),
            'text_raw': row['text'],
            'sample_id': idx
        }
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.text_max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def _process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Process audio input
        """
        full_path = self.data_path / audio_path
        
        if not full_path.exists():
            # Return zero tensor if audio file doesn't exist
            return torch.zeros(self.config.audio_max_length)
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(full_path)
            
            # Resample if necessary
            if sample_rate != self.config.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.config.audio_sample_rate
                )
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or truncate
            if waveform.shape[1] > self.config.audio_max_length:
                waveform = waveform[:, :self.config.audio_max_length]
            else:
                padding = self.config.audio_max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Apply augmentation if enabled
            if self.augment and self.split == 'train':
                waveform = self._augment_audio(waveform)
            
            return waveform.squeeze(0)
            
        except Exception as e:
            print(f"Error loading audio {full_path}: {e}")
            return torch.zeros(self.config.audio_max_length)
    
    def _process_video(self, video_path: str) -> torch.Tensor:
        """
        Process video input
        """
        full_path = self.data_path / video_path
        
        if not full_path.exists():
            # Return zero tensor if video file doesn't exist
            return torch.zeros(
                self.config.video_max_frames, 
                3, 
                *self.config.video_frame_size
            )
        
        try:
            # Load video
            cap = cv2.VideoCapture(str(full_path))
            frames = []
            
            while len(frames) < self.config.video_max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame = cv2.resize(frame, self.config.video_frame_size)
                
                # Convert to tensor and normalize
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC to CHW
                
                frames.append(frame)
            
            cap.release()
            
            # Pad with zeros if not enough frames
            while len(frames) < self.config.video_max_frames:
                frames.append(torch.zeros(3, *self.config.video_frame_size))
            
            video_tensor = torch.stack(frames)
            
            # Apply augmentation if enabled
            if self.augment and self.split == 'train':
                video_tensor = self._augment_video(video_tensor)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video {full_path}: {e}")
            return torch.zeros(
                self.config.video_max_frames, 
                3, 
                *self.config.video_frame_size
            )
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply audio augmentation
        """
        # Add noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(waveform) * 0.01
            waveform = waveform + noise
        
        # Time stretching (simple implementation)
        if torch.rand(1) < 0.3:
            stretch_factor = 0.8 + torch.rand(1).item() * 0.4  # Random between 0.8 and 1.2
            # Simple time stretching by resampling
            original_length = waveform.shape[-1]  # Get the last dimension (time)
            new_length = int(original_length * stretch_factor)
            
            if new_length > 0:
                # Ensure waveform is 2D for interpolation (batch_size=1, channels=1, length)
                if waveform.dim() == 1:
                    waveform_2d = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, length]
                elif waveform.dim() == 2:
                    waveform_2d = waveform.unsqueeze(0)  # [1, channels, length]
                else:
                    waveform_2d = waveform
                
                # Use proper interpolation for audio signal
                stretched = torch.nn.functional.interpolate(
                    waveform_2d,
                    size=new_length,
                    mode='linear',
                    align_corners=False
                )
                
                # Remove batch dimension and convert back to original shape
                if waveform.dim() == 1:
                    waveform = stretched.squeeze(0).squeeze(0)
                elif waveform.dim() == 2:
                    waveform = stretched.squeeze(0)
                else:
                    waveform = stretched
                
                # Pad or truncate to original length
                if new_length > self.config.audio_max_length:
                    waveform = waveform[..., :self.config.audio_max_length]
                else:
                    padding = self.config.audio_max_length - new_length
                    if waveform.dim() == 1:
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
                    else:
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform

    def _augment_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply video augmentation
        """
        # Random brightness adjustment
        if torch.rand(1) < 0.3:
            brightness_factor = 0.8 + torch.rand(1).item() * 0.4  # Random between 0.8 and 1.2
            video = torch.clamp(video * brightness_factor, 0, 1)
        
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            video = torch.flip(video, dims=[3])
        
        return video


class CMUMOSEIDataset(MultimodalDataset):
    """
    CMU-MOSEI dataset loader
    """
    def _load_data(self) -> pd.DataFrame:
        return super()._load_data()


class MELDDataset(MultimodalDataset):
    """
    MELD dataset loader
    """
    def _load_data(self) -> pd.DataFrame:
        return super()._load_data()


class IEMOCAPDataset(MultimodalDataset):
    """
    IEMOCAP dataset loader
    """
    def _load_data(self) -> pd.DataFrame:
        return super()._load_data()


class SamplePDataset(MultimodalDataset):
    """
    Sample dataset loader
    """
    def _load_data(self) -> pd.DataFrame:
        return super()._load_data()


class FewShotDataset(Dataset):
    """
    Few-shot learning dataset wrapper
    """
    def __init__(
        self,
        base_dataset: MultimodalDataset,
        n_shot: int,
        n_way: int = None,
        seed: int = 42
    ):
        self.base_dataset = base_dataset
        self.n_shot = n_shot
        self.n_way = n_way or base_dataset.config.num_emotions
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Sample few-shot examples
        self.few_shot_indices = self._sample_few_shot_indices()
    
    def _sample_few_shot_indices(self) -> List[int]:
        """
        Sample indices for few-shot learning
        """
        indices_by_class = {}
        
        for idx in range(len(self.base_dataset)):
            emotion = self.base_dataset.data.iloc[idx]['emotion']
            emotion_id = self.base_dataset.emotion_to_id[emotion]
            
            if emotion_id not in indices_by_class:
                indices_by_class[emotion_id] = []
            indices_by_class[emotion_id].append(idx)
        
        # Sample n_shot examples per class
        few_shot_indices = []
        for class_id in range(self.n_way):
            if class_id in indices_by_class:
                class_indices = indices_by_class[class_id]
                sampled = np.random.choice(
                    class_indices, 
                    min(self.n_shot, len(class_indices)), 
                    replace=False
                )
                few_shot_indices.extend(sampled)
        
        return few_shot_indices
    
    def __len__(self) -> int:
        return len(self.few_shot_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_idx = self.few_shot_indices[idx]
        return self.base_dataset[original_idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader with custom collate function
    """
    def collate_fn(batch):
        text_input_ids = torch.stack([item['text']['input_ids'] for item in batch])
        text_attention_mask = torch.stack([item['text']['attention_mask'] for item in batch])
        audio = torch.stack([item['audio'] for item in batch])
        video = torch.stack([item['video'] for item in batch])
        emotions = torch.stack([item['emotion'] for item in batch])
        
        return {
            'text': {
                'input_ids': text_input_ids,
                'attention_mask': text_attention_mask
            },
            'audio': audio,
            'video': video,
            'emotion': emotions,
            'text_raw': [item['text_raw'] for item in batch],
            'sample_ids': [item['sample_id'] for item in batch]
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def get_dataset(
    dataset_name: str,
    data_path: str,
    split: str,
    config: 'config.ModelConfig',
    augment: bool = False
) -> MultimodalDataset:
    """
    Factory function to get dataset by name
    """
    dataset_classes = {
        'cmu_mosei': CMUMOSEIDataset,
        'meld': MELDDataset,
        'iemocap': IEMOCAPDataset,
        'multimodal': MultimodalDataset,
        'sample': SamplePDataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name](
        data_path=data_path,
        split=split,
        config=config,
        augment=augment
    )
