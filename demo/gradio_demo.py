import gradio as gr
import torch
import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tempfile
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our models and utilities
import sys
sys.path.append('..')
from models.multimodal_model import MultimodalEmotionModel, load_pretrained_model
from config import ModelConfig
from data.dataset_loaders import MultimodalDataset
from training.advanced_trainer import AdvancedTrainer

class MultimodalEmotionDemo:
    """Interactive demo for multimodal emotion recognition"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = ModelConfig(**config_dict)
        else:
            self.config = ModelConfig()
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_pretrained_model(model_path, self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels and colors
        self.emotion_colors = {
            'happy': '#FFD700',
            'sad': '#4169E1', 
            'angry': '#DC143C',
            'fear': '#9932CC',
            'surprise': '#FF69B4',
            'disgust': '#228B22',
            'neutral': '#808080'
        }
        
        # Initialize conversation history
        self.conversation_history = []
        
        # LLM for response generation (placeholder)
        self.response_generator = EmotionAwareResponseGenerator()
    
    def process_multimodal_input(
        self,
        text_input: str,
        audio_file: Optional[str] = None,
        video_file: Optional[str] = None,
        webcam_video: Optional[str] = None
    ) -> Tuple[Dict, str, str, go.Figure, go.Figure]:
        """Process multimodal input and return emotion analysis"""
        
        try:
            # Use webcam video if provided, otherwise use uploaded video
            video_source = webcam_video if webcam_video else video_file
            
            # Process each modality
            text_features = self._process_text(text_input)
            audio_features = self._process_audio(audio_file)
            video_features = self._process_video(video_source)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    text_input=text_features,
                    audio_input=audio_features,
                    video_input=video_features
                )
            
            # Extract results
            emotion_probs = outputs['emotion_probs'].cpu().numpy()[0]
            predicted_emotion = self.config.emotion_labels[np.argmax(emotion_probs)]
            confidence = float(np.max(emotion_probs))
            
            # Get individual modality contributions
            individual_results = self._get_individual_contributions(outputs)
            
            # Generate emotion analysis
            emotion_analysis = {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'emotion_distribution': {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.config.emotion_labels, emotion_probs)
                },
                'individual_modalities': individual_results,
                'valence': float(outputs.get('valence', torch.tensor([0.0])).cpu().numpy()[0]),
                'arousal': float(outputs.get('arousal', torch.tensor([0.0])).cpu().numpy()[0])
            }
            
            # Generate AI response
            ai_response = self.response_generator.generate_response(
                text_input, predicted_emotion, confidence, emotion_analysis
            )
            
            # Generate activity suggestions
            suggestions = self._generate_activity_suggestions(predicted_emotion, emotion_analysis)
            
            # Create visualizations
            emotion_chart = self._create_emotion_chart(emotion_analysis['emotion_distribution'])
            valence_arousal_chart = self._create_valence_arousal_chart(
                emotion_analysis['valence'], 
                emotion_analysis['arousal'],
                predicted_emotion
            )
            
            # Update conversation history
            self.conversation_history.append({
                'user_input': text_input,
                'emotion': predicted_emotion,
                'confidence': confidence,
                'ai_response': ai_response,
                'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
            })
            
            return (
                emotion_analysis,
                ai_response,
                suggestions,
                emotion_chart,
                valence_arousal_chart
            )
            
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            return {}, error_msg, "Please try again with valid inputs.", None, None
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input"""
        if not text.strip():
            text = "No text provided"
        
        # Tokenize text
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name)
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.text_max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def _process_audio(self, audio_file: Optional[str]) -> torch.Tensor:
        """Process audio input"""
        if not audio_file:
            # Return zero tensor if no audio
            return torch.zeros(1, self.config.audio_max_length).to(self.device)
        
        try:
            # Load audio file
            waveform, sample_rate = librosa.load(audio_file, sr=self.config.audio_sample_rate)
            
            # Convert to tensor
            waveform = torch.from_numpy(waveform).float()
            
            # Pad or truncate
            if len(waveform) > self.config.audio_max_length:
                waveform = waveform[:self.config.audio_max_length]
            else:
                padding = self.config.audio_max_length - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return torch.zeros(1, self.config.audio_max_length).to(self.device)
    
    def _process_video(self, video_file: Optional[str]) -> torch.Tensor:
        """Process video input"""
        if not video_file:
            # Return zero tensor if no video
            return torch.zeros(
                1, self.config.video_max_frames, 3, 
                *self.config.video_frame_size
            ).to(self.device)
        
        try:
            # Load video
            cap = cv2.VideoCapture(video_file)
            frames = []
            
            frame_count = 0
            while len(frames) < self.config.video_max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to get roughly video_max_frames frames
                if frame_count % max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.config.video_max_frames) == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame
                    frame = cv2.resize(frame, self.config.video_frame_size)
                    
                    # Convert to tensor and normalize
                    frame = torch.from_numpy(frame).float() / 255.0
                    frame = frame.permute(2, 0, 1)  # HWC to CHW
                    
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Pad with zeros if not enough frames
            while len(frames) < self.config.video_max_frames:
                frames.append(torch.zeros(3, *self.config.video_frame_size))
            
            video_tensor = torch.stack(frames).unsqueeze(0)
            return video_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return torch.zeros(
                1, self.config.video_max_frames, 3, 
                *self.config.video_frame_size
            ).to(self.device)
    
    def _get_individual_contributions(self, outputs: Dict) -> Dict[str, Dict]:
        """Get individual modality contributions"""
        individual_results = {}
        
        # If we have individual logits (from late fusion)
        if 'individual_logits' in outputs:
            for modality, logits in outputs['individual_logits'].items():
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                predicted = self.config.emotion_labels[np.argmax(probs)]
                confidence = float(np.max(probs))
                
                individual_results[modality] = {
                    'predicted_emotion': predicted,
                    'confidence': confidence,
                    'distribution': {
                        emotion: float(prob) 
                        for emotion, prob in zip(self.config.emotion_labels, probs)
                    }
                }
        
        return individual_results
    
    def _create_emotion_chart(self, emotion_distribution: Dict[str, float]) -> go.Figure:
        """Create emotion distribution chart"""
        emotions = list(emotion_distribution.keys())
        probabilities = list(emotion_distribution.values())
        colors = [self.emotion_colors.get(emotion, '#808080') for emotion in emotions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=emotions,
                y=probabilities,
                marker_color=colors,
                text=[f'{p:.2%}' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Emotion Distribution',
            xaxis_title='Emotions',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_valence_arousal_chart(
        self, 
        valence: float, 
        arousal: float, 
        emotion: str
    ) -> go.Figure:
        """Create valence-arousal 2D chart"""
        
        # Emotion positions in valence-arousal space (approximate)
        emotion_positions = {
            'happy': (0.8, 0.6),
            'surprise': (0.4, 0.8),
            'angry': (-0.6, 0.7),
            'fear': (-0.6, 0.8),
            'sad': (-0.7, -0.4),
            'disgust': (-0.6, 0.2),
            'neutral': (0.0, 0.0)
        }
        
        fig = go.Figure()
        
        # Add all emotion positions as reference
        for emo, (v, a) in emotion_positions.items():
            fig.add_trace(go.Scatter(
                x=[v], y=[a],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=self.emotion_colors.get(emo, '#808080'),
                    opacity=0.6
                ),
                text=[emo],
                textposition='top center',
                name=emo,
                showlegend=False
            ))
        
        # Add current prediction
        fig.add_trace(go.Scatter(
            x=[valence], y=[arousal],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name='Current Prediction',
            text=[f'Predicted: {emotion}'],
            textposition='bottom center'
        ))
        
        fig.update_layout(
            title='Valence-Arousal Space',
            xaxis_title='Valence (Negative ← → Positive)',
            yaxis_title='Arousal (Low ← → High)',
            xaxis=dict(range=[-1, 1], zeroline=True),
            yaxis=dict(range=[-1, 1], zeroline=True),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _generate_activity_suggestions(
        self, 
        emotion: str, 
        emotion_analysis: Dict
    ) -> str:
        """Generate activity suggestions based on detected emotion"""
        
        suggestions_map = {
            'happy': [
                "🎉 Share your joy with friends and family!",
                "📸 Capture this moment with photos or journaling",
                "🎵 Listen to upbeat music or dance",
                "🌟 Use this positive energy for creative activities",
                "💝 Do something kind for others"
            ],
            'sad': [
                "🤗 Reach out to a trusted friend or family member",
                "📖 Try reading a comforting book or watching uplifting content",
                "🚶‍♀️ Take a gentle walk in nature",
                "🎨 Express yourself through art or writing",
                "☕ Make yourself a warm drink and practice self-care"
            ],
            'angry': [
                "🧘‍♀️ Try deep breathing or meditation exercises",
                "🏃‍♂️ Channel energy into physical exercise",
                "📝 Write down your feelings to process them",
                "🎵 Listen to calming music",
                "💬 Talk to someone you trust about what's bothering you"
            ],
            'fear': [
                "🛡️ Focus on what you can control in the situation",
                "🧘‍♀️ Practice grounding techniques (5-4-3-2-1 method)",
                "💪 Break down the challenge into smaller, manageable steps",
                "🤝 Seek support from friends, family, or professionals",
                "📚 Learn more about what's causing the fear"
            ],
            'surprise': [
                "🤔 Take a moment to process what just happened",
                "📝 Write down your thoughts and reactions",
                "💬 Share the experience with someone close to you",
                "🎯 Consider how this might change your plans or perspective",
                "🌟 Embrace the unexpected as an opportunity for growth"
            ],
            'disgust': [
                "🚿 Remove yourself from the unpleasant situation if possible",
                "🧘‍♀️ Practice mindfulness to observe the feeling without judgment",
                "🌿 Engage in cleansing activities (shower, clean space)",
                "💭 Reflect on what values this reaction might be protecting",
                "🎯 Focus on positive alternatives or solutions"
            ],
            'neutral': [
                "🎯 This might be a good time to set new goals",
                "📚 Learn something new or pick up a hobby",
                "🤝 Connect with friends or family",
                "🌟 Try a new experience or activity",
                "🧘‍♀️ Practice mindfulness or meditation"
            ]
        }
        
        suggestions = suggestions_map.get(emotion, suggestions_map['neutral'])
        
        # Add confidence-based message
        confidence = emotion_analysis['confidence']
        if confidence > 0.8:
            confidence_msg = f"I'm quite confident (${confidence:.1%}) that you're feeling {emotion}."
        elif confidence > 0.6:
            confidence_msg = f"I think (${confidence:.1%} confidence) you might be feeling {emotion}."
        else:
            confidence_msg = f"I'm not entirely sure, but you might be feeling {emotion} (${confidence:.1%} confidence)."
        
        # Format suggestions
        suggestion_text = f"{confidence_msg}\n\nHere are some suggestions:\n\n"
        suggestion_text += "\n".join(suggestions[:3])  # Show top 3 suggestions
        
        return suggestion_text


class EmotionAwareResponseGenerator:
    """Generate contextually appropriate responses based on detected emotions"""
    
    def __init__(self):
        self.response_templates = {
            'happy': [
                "That's wonderful to hear! Your happiness is really coming through. {context}",
                "I can sense your joy! It's great that you're feeling so positive. {context}",
                "Your enthusiasm is contagious! {context}"
            ],
            'sad': [
                "I can hear that you're going through a tough time. {context} Remember that it's okay to feel this way.",
                "It sounds like you're feeling down. {context} Your feelings are valid, and I'm here to listen.",
                "I sense some sadness in what you're sharing. {context} Would you like to talk about it more?"
            ],
            'angry': [
                "I can tell you're feeling frustrated about this. {context} It's understandable to feel angry sometimes.",
                "Your anger comes through clearly. {context} Let's see if we can work through this together.",
                "I hear the frustration in your message. {context} What do you think might help right now?"
            ],
            'fear': [
                "I can sense some worry or anxiety in what you're saying. {context} It's natural to feel scared sometimes.",
                "It sounds like you might be feeling anxious about this. {context} You're not alone in feeling this way.",
                "I detect some concern in your message. {context} Would it help to talk through what's worrying you?"
            ],
            'surprise': [
                "Wow, that sounds unexpected! {context} How are you processing this surprise?",
                "That must have caught you off guard! {context} Surprises can be quite overwhelming.",
                "I can sense your amazement! {context} What a surprising turn of events!"
            ],
            'disgust': [
                "I can tell something is really bothering you. {context} It's okay to feel repulsed by certain things.",
                "That sounds quite unpleasant. {context} Your reaction is completely understandable.",
                "I sense your strong negative reaction. {context} Sometimes we encounter things that just don't sit right with us."
            ],
            'neutral': [
                "Thanks for sharing that with me. {context} How can I help you today?",
                "I appreciate you telling me about this. {context} What would you like to explore further?",
                "Interesting perspective. {context} What are your thoughts on this?"
            ]
        }
    
    def generate_response(
        self, 
        user_input: str, 
        emotion: str, 
        confidence: float, 
        emotion_analysis: Dict
    ) -> str:
        """Generate an appropriate response based on emotion and context"""
        
        # Get response template
        templates = self.response_templates.get(emotion, self.response_templates['neutral'])
        template = np.random.choice(templates)
        
        # Generate context-aware addition
        context = self._generate_context(user_input, emotion, confidence)
        
        # Fill template
        response = template.format(context=context)
        
        # Add follow-up question based on emotion
        follow_up = self._generate_follow_up(emotion, confidence)
        if follow_up:
            response += f" {follow_up}"
        
        return response
    
    def _generate_context(self, user_input: str, emotion: str, confidence: float) -> str:
        """Generate contextual information"""
        
        # Simple keyword-based context generation
        keywords = user_input.lower().split()
        
        if any(word in keywords for word in ['work', 'job', 'boss', 'colleague']):
            return "Work situations can really affect our emotions."
        elif any(word in keywords for word in ['family', 'parent', 'child', 'sibling']):
            return "Family relationships are so important to our wellbeing."
        elif any(word in keywords for word in ['friend', 'friendship']):
            return "Friendships play such a vital role in our lives."
        elif any(word in keywords for word in ['school', 'study', 'exam', 'test']):
            return "Academic pressures can be quite intense."
        else:
            return "Life has its ups and downs."
    
    def _generate_follow_up(self, emotion: str, confidence: float) -> str:
        """Generate appropriate follow-up questions"""
        
        if confidence < 0.6:
            return "I'm not entirely sure I've understood your emotional state correctly. How are you really feeling?"
        
        follow_ups = {
            'happy': "What's been the highlight of your day?",
            'sad': "Is there anything specific that's been weighing on your mind?",
            'angry': "What do you think would help you feel better right now?",
            'fear': "Would you like to talk about what's making you feel anxious?",
            'surprise': "How do you think this will change things for you?",
            'disgust': "What would help you move past this unpleasant feeling?",
            'neutral': "What's on your mind today?"
        }
        
        return follow_ups.get(emotion, "How can I support you better?")


def create_gradio_interface(model_path: str) -> gr.Interface:
    """Create the Gradio interface"""
    
    # Initialize demo
    demo_system = MultimodalEmotionDemo(model_path)
    
    def process_input(text, audio, video, webcam):
        """Process all inputs and return results"""
        try:
            results = demo_system.process_multimodal_input(text, audio, video, webcam)
            emotion_analysis, ai_response, suggestions, emotion_chart, va_chart = results
            
            # Format emotion analysis for display
            if emotion_analysis:
                analysis_text = f"""
                **Predicted Emotion:** {emotion_analysis['predicted_emotion']} 
                **Confidence:** {emotion_analysis['confidence']:.1%}
                **Valence:** {emotion_analysis['valence']:.2f} (Negative ← → Positive)
                **Arousal:** {emotion_analysis['arousal']:.2f} (Low ← → High)
                
                **Emotion Breakdown:**
                """
                
                for emotion, prob in emotion_analysis['emotion_distribution'].items():
                    analysis_text += f"\n• {emotion.title()}: {prob:.1%}"
                
                # Add individual modality results if available
                if emotion_analysis['individual_modalities']:
                    analysis_text += "\n\n**Individual Modality Results:**"
                    for modality, result in emotion_analysis['individual_modalities'].items():
                        analysis_text += f"\n• {modality.title()}: {result['predicted_emotion']} ({result['confidence']:.1%})"
            else:
                analysis_text = "Error in emotion analysis"
            
            return analysis_text, ai_response, suggestions, emotion_chart, va_chart
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg, error_msg, "Please try again", None, None
    
    # Create interface
    with gr.Blocks(title="Multimodal Emotion Recognition System", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎭 Multimodal Emotion Recognition & Interactive Dialogue System
        
        This system analyzes your emotions from **text**, **audio**, and **video** inputs, then provides personalized responses and activity suggestions.
        
        ### How to use:
        1. **Text**: Type what you're thinking or feeling
        2. **Audio**: Upload an audio file or record your voice
        3. **Video**: Upload a video file or use your webcam
        4. Click "Analyze Emotions" to get results
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Your Data")
                
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Tell me how you're feeling or what's on your mind...",
                    lines=3
                )
                
                audio_input = gr.Audio(
                    label="Audio Input (Optional)",
                    type="filepath"
                )
                
                video_input = gr.Video(
                    label="Video Upload (Optional)",
                )
                
                webcam_input = gr.Video(
                    label="Webcam Recording (Optional)",
                    source="webcam"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Emotions", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Emotion Analysis Results")
                
                with gr.Row():
                    emotion_analysis_output = gr.Textbox(
                        label="Detailed Analysis",
                        lines=10,
                        interactive=False
                    )
                    
                with gr.Row():
                    emotion_chart = gr.Plot(label="Emotion Distribution")
                    valence_arousal_chart = gr.Plot(label="Valence-Arousal Space")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 AI Response")
                ai_response_output = gr.Textbox(
                    label="Personalized Response",
                    lines=4,
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("### 💡 Activity Suggestions")
                suggestions_output = gr.Textbox(
                    label="Recommended Activities",
                    lines=4,
                    interactive=False
                )
        
        # Connect the analyze button to the processing function
        analyze_btn.click(
            fn=process_input,
            inputs=[text_input, audio_input, video_input, webcam_input],
            outputs=[
                emotion_analysis_output, 
                ai_response_output, 
                suggestions_output,
                emotion_chart,
                valence_arousal_chart
            ]
        )
        
        # Add examples
        gr.Markdown("### 📋 Example Inputs")
        
        examples = [
            ["I'm so excited about my vacation next week! I can't wait to relax on the beach.", None, None, None],
            ["I've been feeling really stressed about work lately. My boss keeps piling on more tasks.", None, None, None],
            ["I can't believe I got the promotion! This is the best day ever!", None, None, None],
            ["I'm worried about the presentation tomorrow. What if I mess up?", None, None, None],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[text_input, audio_input, video_input, webcam_input],
            outputs=[
                emotion_analysis_output, 
                ai_response_output, 
                suggestions_output,
                emotion_chart,
                valence_arousal_chart
            ],
            fn=process_input,
            cache_examples=False
        )
        
        gr.Markdown("""
        ### 🔬 Technical Details
        
        This system uses:
        - **Text Analysis**: DeBERTa transformer for natural language understanding
        - **Audio Analysis**: Wav2Vec2 for speech emotion recognition  
        - **Video Analysis**: Vision Transformer (ViT) for facial expression analysis
        - **Fusion**: Advanced multimodal fusion with attention mechanisms
        - **Response Generation**: Emotion-aware dialogue system
        
        **Supported Emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
        """)
    
    return interface


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Multimodal Emotion Recognition Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model configuration file")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the demo on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = create_gradio_interface(args.model_path)
    interface.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )
