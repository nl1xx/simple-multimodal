"""
Sample Data Generator for Multimodal Emotion Recognition

This script creates sample data to help users understand the expected data format
and test the system before using real datasets.
"""
import cv2
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
from typing import List, Tuple


def create_sample_audio(
    emotion: str, 
    duration: float = 3.0, 
    sample_rate: int = 16000,
    output_path: str = None
) -> str:
    """
    Create sample audio file for given emotion
    """
    
    # Generate different audio patterns for different emotions
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    if emotion == 'happy':
        # Higher frequency, more variation
        audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        audio += 0.1 * np.random.randn(len(t))  # Add some noise
    elif emotion == 'sad':
        # Lower frequency, slower
        audio = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 110 * t)
        audio *= np.exp(-t * 0.5)  # Decay
    elif emotion == 'angry':
        # Harsh, noisy
        audio = 0.5 * np.sin(2 * np.pi * 330 * t)
        audio += 0.3 * np.random.randn(len(t))  # More noise
    elif emotion == 'fear':
        # Trembling effect
        tremolo = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
        audio = 0.3 * np.sin(2 * np.pi * 400 * t) * tremolo
    elif emotion == 'surprise':
        # Sudden burst
        audio = np.zeros_like(t)
        burst_start = int(len(t) * 0.3)
        burst_end = int(len(t) * 0.7)
        audio[burst_start:burst_end] = 0.6 * np.sin(2 * np.pi * 600 * t[burst_start:burst_end])
    elif emotion == 'disgust':
        # Low, rough
        audio = 0.4 * np.sin(2 * np.pi * 150 * t)
        audio += 0.2 * np.sin(2 * np.pi * 75 * t)
    else:  # neutral
        # Simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save audio file
    if output_path is None:
        output_path = f"sample_{emotion}.wav"
    
    sf.write(output_path, audio, sample_rate)
    return output_path


def create_sample_video(
    emotion: str,
    duration: float = 3.0,
    fps: int = 15,
    size: Tuple[int, int] = (224, 224),
    output_path: str = None
) -> str:
    """
    Create sample video file for given emotion
    """
    
    if output_path is None:
        output_path = f"sample_{emotion}.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create a frame based on emotion
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Different visual patterns for different emotions
        if emotion == 'happy':
            # Bright yellow/orange colors
            color = (0, 165, 255)  # Orange in BGR
            center = (size[0]//2, size[1]//2)
            radius = int(50 + 20 * np.sin(frame_idx * 0.3))
            cv2.circle(frame, center, radius, color, -1)
            
        elif emotion == 'sad':
            # Blue colors, drooping pattern
            color = (255, 100, 0)  # Blue in BGR
            points = np.array([
                [size[0]//4, size[1]//3],
                [size[0]//2, size[1]//2 + 20],
                [3*size[0]//4, size[1]//3]
            ], np.int32)
            cv2.fillPoly(frame, [points], color)
            
        elif emotion == 'angry':
            # Red colors, sharp patterns
            color = (0, 0, 255)  # Red in BGR
            # Draw zigzag pattern
            for i in range(0, size[0], 20):
                y1 = size[1]//3 if (i//20) % 2 == 0 else 2*size[1]//3
                y2 = 2*size[1]//3 if (i//20) % 2 == 0 else size[1]//3
                cv2.line(frame, (i, y1), (i+20, y2), color, 5)
                
        elif emotion == 'fear':
            # Dark purple, shaky pattern
            color = (128, 0, 128)  # Purple in BGR
            # Add random noise to simulate shaking
            noise_x = int(10 * np.random.randn())
            noise_y = int(10 * np.random.randn())
            center = (size[0]//2 + noise_x, size[1]//2 + noise_y)
            cv2.circle(frame, center, 30, color, -1)
            
        elif emotion == 'surprise':
            # Bright white/yellow, expanding pattern
            color = (255, 255, 255)  # White in BGR
            radius = int(20 + frame_idx * 2)
            if radius > 100:
                radius = 100
            center = (size[0]//2, size[1]//2)
            cv2.circle(frame, center, radius, color, 3)
            
        elif emotion == 'disgust':
            # Green colors, wavy pattern
            color = (0, 255, 0)  # Green in BGR
            for x in range(0, size[0], 5):
                y = int(size[1]//2 + 30 * np.sin(x * 0.1 + frame_idx * 0.2))
                cv2.circle(frame, (x, y), 3, color, -1)
                
        else:  # neutral
            # Gray, simple pattern
            color = (128, 128, 128)  # Gray in BGR
            cv2.rectangle(frame, (size[0]//4, size[1]//4), (3*size[0]//4, 3*size[1]//4), color, 2)
        
        out.write(frame)
    
    out.release()
    return output_path


def create_sample_dataset(
    output_dir: str = "data/sample",
    num_samples_per_emotion: int = 10,
    emotions: List[str] = None
):
    """
    Create a complete sample dataset
    """
    
    if emotions is None:
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    
    # Create directories
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio'
    video_dir = output_path / 'video'
    
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)
    
    # Sample texts for each emotion
    sample_texts = {
        'happy': [
            "I'm so excited about this!",
            "This is the best day ever!",
            "I feel absolutely wonderful!",
            "Everything is going perfectly!",
            "I can't stop smiling!",
            "This makes me so happy!",
            "I'm thrilled about the news!",
            "Life is beautiful today!",
            "I'm overjoyed with the results!",
            "This brings me so much joy!"
        ],
        'sad': [
            "I feel really down today.",
            "This makes me so sad.",
            "I'm feeling quite depressed.",
            "Everything seems hopeless.",
            "I can't stop feeling blue.",
            "This is really disappointing.",
            "I feel like crying.",
            "My heart feels heavy.",
            "I'm going through a tough time.",
            "This news really upsets me."
        ],
        'angry': [
            "This is absolutely infuriating!",
            "I'm so mad about this!",
            "This makes my blood boil!",
            "I can't believe this happened!",
            "This is completely unacceptable!",
            "I'm furious right now!",
            "This is driving me crazy!",
            "I'm really ticked off!",
            "This is so frustrating!",
            "I'm livid about this situation!"
        ],
        'fear': [
            "I'm really scared about this.",
            "This makes me very anxious.",
            "I'm worried something bad will happen.",
            "This terrifies me completely.",
            "I feel so nervous and afraid.",
            "This gives me the chills.",
            "I'm trembling with fear.",
            "This is my worst nightmare.",
            "I'm panicking about the outcome.",
            "This fills me with dread."
        ],
        'surprise': [
            "Wow, I didn't expect that!",
            "This is so surprising!",
            "I can't believe my eyes!",
            "What a shocking revelation!",
            "This caught me off guard!",
            "I'm absolutely amazed!",
            "This is incredible!",
            "I never saw this coming!",
            "What a pleasant surprise!",
            "This is mind-blowing!"
        ],
        'disgust': [
            "This is absolutely revolting.",
            "I find this really disgusting.",
            "This makes me feel sick.",
            "This is completely repulsive.",
            "I can't stand this at all.",
            "This is so gross and nasty.",
            "This makes my stomach turn.",
            "I'm repelled by this behavior.",
            "This is utterly distasteful.",
            "This disgusts me to my core."
        ],
        'neutral': [
            "This is a normal day.",
            "Everything seems ordinary.",
            "Nothing special is happening.",
            "This is just a regular occurrence.",
            "I'm feeling pretty neutral about this.",
            "This is neither good nor bad.",
            "It's just another typical situation.",
            "I have no strong feelings about this.",
            "This is quite unremarkable.",
            "Everything is proceeding as usual."
        ]
    }
    
    # Generate data for each split
    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test
    
    all_data = []
    sample_id = 0
    
    print("Generating sample dataset...")
    
    for emotion in emotions:
        print(f"Creating samples for emotion: {emotion}")
        
        for i in range(num_samples_per_emotion):
            # Create audio file
            audio_filename = f"{emotion}_{i:03d}.wav"
            audio_path = audio_dir / audio_filename
            create_sample_audio(emotion, output_path=str(audio_path))
            
            # Create video file
            video_filename = f"{emotion}_{i:03d}.mp4"
            video_path = video_dir / video_filename
            create_sample_video(emotion, output_path=str(video_path))
            
            # Get sample text
            text = sample_texts[emotion][i % len(sample_texts[emotion])]
            
            # Add to data
            all_data.append({
                'text': text,
                'audio_path': f'audio/{audio_filename}',
                'video_path': f'video/{video_filename}',
                'emotion': emotion,
                'sample_id': sample_id
            })
            
            sample_id += 1
    
    # Shuffle data
    np.random.shuffle(all_data)
    
    # Split data
    total_samples = len(all_data)
    train_end = int(total_samples * split_ratios[0])
    val_end = train_end + int(total_samples * split_ratios[1])
    
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    
    # Save CSV files
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        df = pd.DataFrame(split_data)
        csv_path = output_path / f'{split_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"Created {split_name}.csv with {len(split_data)} samples")
    
    print(f"Sample dataset created successfully in: {output_path}")
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create sample multimodal emotion dataset")
    
    parser.add_argument('--output_dir', type=str, default='data/sample', help='Output directory for sample dataset')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per emotion')
    parser.add_argument('--emotions', nargs='+', default=['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'],
                       help='List of emotions to generate')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample dataset
    dataset_path = create_sample_dataset(
        output_dir=args.output_dir,
        num_samples_per_emotion=args.num_samples,
        emotions=args.emotions
    )
    
    print("Sample dataset ready!")
    print("Location: {dataset_path}")
    print("You can now test the system with:")
    print(f"python train_advanced.py --data_path {dataset_path} --epochs 5")


if __name__ == "__main__":
    main()
