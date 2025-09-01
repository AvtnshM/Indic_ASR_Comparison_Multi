import gradio as gr
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCTC
import librosa
import numpy as np
from jiwer import wer, cer
import time

# Model configurations
MODEL_CONFIGS = {
    "AudioX-North (Jivi AI)": {
        "repo": "jiviai/audioX-north-v1",
        "model_type": "seq2seq",
        "description": "Supports Hindi, Gujarati, Marathi"
    },
    "IndicConformer (AI4Bharat)": {
        "repo": "ai4bharat/indic-conformer-600m-multilingual",
        "model_type": "ctc_rnnt",
        "description": "Supports 22 Indian languages"
    },
    "MMS (Facebook)": {
        "repo": "facebook/mms-1b",
        "model_type": "ctc",
        "description": "Supports over 1,400 languages (fine-tuning recommended)"
    }
}

# Load model and processor
def load_model_and_processor(model_name):
    config = MODEL_CONFIGS[model_name]
    repo = config["repo"]
    model_type = config["model_type"]
    
    try:
        processor = AutoProcessor.from_pretrained(repo)
        if model_type == "seq2seq":
            model = AutoModelForSpeechSeq2Seq.from_pretrained(repo)
        else:  # ctc or ctc_rnnt
            model = AutoModelForCTC.from_pretrained(repo)
        return model, processor, model_type
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Compute metrics (WER, CER, RTF)
def compute_metrics(reference, hypothesis, audio_duration):
    if not reference or not hypothesis:
        return None, None, None
    try:
        wer_score = wer(reference, hypothesis)
        cer_score = cer(reference, hypothesis)
        rtf = audio_duration / time.time()  # Simplified; actual RTF needs processing time
        return wer_score, cer_score, rtf
    except Exception as e:
        return None, None, f"Error computing metrics: {str(e)}"

def transcribe_audio(audio_file, model_name, reference_text=""):
    if not audio_file:
        return "Please upload an audio file.", None, None, None
    
    # Load model and processor
    model, processor, model_type = load_model_and_processor(model_name)
    if isinstance(model_type, str) and model_type.startswith("Error"):
        return model_type, None, None, None
    
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_file, sr=16000)
        audio_duration = len(audio) / sr
        
        # Process audio
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        
        # Measure processing time for RTF
        start_time = time.time()
        with torch.no_grad():
            if model_type == "seq2seq":
                outputs = model.generate(input_features)
            else:  # ctc or ctc_rnnt
                outputs = model(input_features).logits
                outputs = torch.argmax(outputs, dim=-1)
        
        # Decode transcription
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Compute metrics if reference text is provided
        wer_score, cer_score, rtf = None, None, None
        if reference_text:
            wer_score, cer_score, rtf_error = compute_metrics(reference_text, transcription, audio_duration)
            if isinstance(rtf_error, str):
                return transcription, wer_score, cer_score, rtf_error
            rtf = (time.time() - start_time) / audio_duration  # Actual RTF
        
        return transcription, wer_score, cer_score, rtf
    except Exception as e:
        return f"Error during transcription: {str(e)}", None, None, None

# Gradio interface
def create_interface():
    model_choices = list(MODEL_CONFIGS.keys())
    return gr.Interface(
        fn=transcribe_audio,
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio File (16kHz recommended)"),
            gr.Dropdown(choices=model_choices, label="Select Model", value=model_choices[0]),
            gr.Textbox(label="Reference Text (Optional for WER/CER)", placeholder="Enter ground truth text here")
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.Textbox(label="WER"),
            gr.Textbox(label="CER"),
            gr.Textbox(label="RTF")
        ],
        title="Multilingual Speech-to-Text with Metrics",
        description="Upload an audio file, select a model, and optionally provide reference text to compute WER, CER, and RTF.",
        allow_flagging="never"
    )

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()