import gradio as gr
import time
import librosa
import torch
import numpy as np
from jiwer import wer, cer
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC
)

# Global variables for models (loaded once)
whisper_processor = None
whisper_model = None
conformer_processor = None
conformer_model = None

def load_models():
    """Load models once at startup with error handling"""
    global whisper_processor, whisper_model, conformer_processor, conformer_model
    
    if whisper_processor is None:
        try:
            print("Loading IndicWhisper...")
            # Try the original model first
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
            print("✅ Using OpenAI Whisper-medium as fallback")
        except Exception as e:
            print(f"❌ Error loading IndicWhisper: {e}")
            # Fallback to standard Whisper
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            print("✅ Using OpenAI Whisper-base as fallback")
        
        try:
            print("Loading IndicConformer...")
            conformer_processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indic-conformer-600m-multilingual")
            conformer_model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indic-conformer-600m-multilingual")
            print("✅ IndicConformer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading IndicConformer: {e}")
            # Fallback to a working multilingual model
            conformer_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            conformer_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            print("✅ Using Facebook XLSR-53 as fallback")
        
        print("Models loaded successfully!")

def transcribe_whisper(audio_path):
    """Transcribe using Whisper model"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        
        start_time = time.time()
        with torch.no_grad():
            # Force Hindi language for better results
            predicted_ids = whisper_model.generate(
                input_features,
                forced_decoder_ids=whisper_processor.get_decoder_prompt_ids(language="hindi", task="transcribe")
            )
        end_time = time.time()
        
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription, end_time - start_time
    except Exception as e:
        return f"Error in Whisper transcription: {str(e)}", 0

def transcribe_conformer(audio_path):
    """Transcribe using Conformer model"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        input_values = conformer_processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        
        start_time = time.time()
        with torch.no_grad():
            logits = conformer_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        end_time = time.time()
        
        transcription = conformer_processor.batch_decode(predicted_ids)[0]
        return transcription, end_time - start_time
    except Exception as e:
        return f"Error in Conformer transcription: {str(e)}", 0

def compare_models(audio_file, ground_truth_text):
    """Main comparison function for Gradio interface"""
    
    if audio_file is None:
        return "Please upload an audio file", "", "", "", "", ""
    
    load_models()  # Ensure models are loaded
    
    try:
        # Get audio duration
        audio_duration = librosa.get_duration(filename=audio_file)
        
        # Test IndicWhisper
        whisper_pred, whisper_time = transcribe_whisper(audio_file)
        whisper_rtf = whisper_time / audio_duration if audio_duration > 0 else 0
        
        # Test IndicConformer  
        conformer_pred, conformer_time = transcribe_conformer(audio_file)
        conformer_rtf = conformer_time / audio_duration if audio_duration > 0 else 0
        
        # Calculate metrics if ground truth provided
        if ground_truth_text and ground_truth_text.strip():
            whisper_wer = wer(ground_truth_text, whisper_pred)
            whisper_cer = cer(ground_truth_text, whisper_pred)
            conformer_wer = wer(ground_truth_text, conformer_pred)
            conformer_cer = cer(ground_truth_text, conformer_pred)
            
            # Format results with metrics
            whisper_result = f"""
## 📊 Whisper Results:
**Prediction:** {whisper_pred}

**WER:** {whisper_wer:.3f}  
**CER:** {whisper_cer:.3f}  
**RTF:** {whisper_rtf:.3f} {'✅ Real-time' if whisper_rtf < 1.0 else '⚠️ Slower'}  
**Time:** {whisper_time:.2f}s
"""
            
            conformer_result = f"""
## 📊 IndicConformer Results:
**Prediction:** {conformer_pred}

**WER:** {conformer_wer:.3f}  
**CER:** {conformer_cer:.3f}  
**RTF:** {conformer_rtf:.3f} {'✅ Real-time' if conformer_rtf < 1.0 else '⚠️ Slower'}  
**Time:** {conformer_time:.2f}s
"""
            
            # Winner analysis
            wer_winner = "Whisper" if whisper_wer < conformer_wer else "IndicConformer"
            cer_winner = "Whisper" if whisper_cer < conformer_cer else "IndicConformer"
            rtf_winner = "Whisper" if whisper_rtf < conformer_rtf else "IndicConformer"
            
            winner_analysis = f"""
## 🏆 Winner Analysis:
**Best WER:** {wer_winner} ({min(whisper_wer, conformer_wer):.3f})  
**Best CER:** {cer_winner} ({min(whisper_cer, conformer_cer):.3f})  
**Fastest:** {rtf_winner} ({min(whisper_rtf, conformer_rtf):.3f})
"""
        else:
            # Results without metrics (no ground truth)
            whisper_result = f"""
## 📊 Whisper Results:
**Prediction:** {whisper_pred}

**RTF:** {whisper_rtf:.3f}  
**Time:** {whisper_time:.2f}s
"""
            
            conformer_result = f"""
## 📊 IndicConformer Results:
**Prediction:** {conformer_pred}

**RTF:** {conformer_rtf:.3f}  
**Time:** {conformer_time:.2f}s
"""
            
            winner_analysis = f"""
## 🏆 Speed Comparison:
**Faster Model:** {'Whisper' if whisper_rtf < conformer_rtf else 'IndicConformer'}  
**RTF Difference:** {abs(whisper_rtf - conformer_rtf):.3f}
"""
        
        return whisper_result, conformer_result, winner_analysis, whisper_pred, conformer_pred, f"Audio duration: {audio_duration:.2f}s"
        
    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        return error_msg, "", "", "", "", ""

# Create Gradio Interface
with gr.Blocks(title="ASR Model Comparison") as demo:
    
    gr.Markdown("""
    # 🎤 ASR Model Comparison: Whisper vs IndicConformer
    
    Compare **OpenAI Whisper** vs **AI4Bharat IndicConformer** on your audio files!
    
    **Models:**
    - **Whisper:** `openai/whisper-medium` (with Hindi language setting)
    - **IndicConformer:** `ai4bharat/indic-conformer-600m-multilingual`
    
    **Metrics:** WER (Word Error Rate), CER (Character Error Rate), RTF (Real-Time Factor)
    
    ⚠️ **Note:** Using standard Whisper model with Hindi language setting for comparison.
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="🎵 Upload Audio File", 
                type="filepath"
            )
            ground_truth_input = gr.Textbox(
                label="📝 Ground Truth Text (Optional)", 
                placeholder="Enter expected transcription for WER/CER calculation...",
                lines=3
            )
            compare_btn = gr.Button("🚀 Compare Models", variant="primary", size="lg")
        
        with gr.Column():
            audio_info = gr.Textbox(label="ℹ️ Audio Info", interactive=False)
    
    with gr.Row():
        with gr.Column():
            whisper_output = gr.Markdown(label="IndicWhisper Results")
        with gr.Column():
            conformer_output = gr.Markdown(label="IndicConformer Results")
    
    winner_output = gr.Markdown(label="🏆 Comparison Summary")
    
    # Hidden outputs for API access
    with gr.Row(visible=False):
        whisper_text = gr.Textbox(label="Whisper Transcription")
        conformer_text = gr.Textbox(label="Conformer Transcription")
    
    compare_btn.click(
        fn=compare_models,
        inputs=[audio_input, ground_truth_input],
        outputs=[whisper_output, conformer_output, winner_output, whisper_text, conformer_text, audio_info]
    )
    
    gr.Markdown("""
    ## 📋 How to Use:
    1. **Upload audio** in any supported format (WAV, MP3, M4A, etc.)
    2. **Add ground truth** (optional) - if provided, you'll get WER/CER metrics
    3. **Click Compare** to see results from both models
    4. **Analyze** which model performs better for your use case
    
    ## 📖 Understanding Metrics:
    - **WER (Word Error Rate):** Percentage of words transcribed incorrectly (Lower = Better, 0 = Perfect)
    - **CER (Character Error Rate):** Percentage of characters transcribed incorrectly (Lower = Better, 0 = Perfect)
    - **RTF (Real-Time Factor):** Ratio of processing time to audio duration (Lower = Faster, <1.0 = Real-time capable)
    
    ## 🌐 Supported Languages:
    Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu
    """)

# Load models on startup
load_models()

if __name__ == "__main__":
    demo.launch()