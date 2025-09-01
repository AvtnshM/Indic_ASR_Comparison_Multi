import gradio as gr
import torch
import torchaudio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCTC,
    AutoModel,
)
import librosa
import numpy as np
from jiwer import wer, cer
import time

# Model configurations
MODEL_CONFIGS = {
    "AudioX-North (Jivi AI)": {
        "repo": "jiviai/audioX-north-v1",
        "model_type": "seq2seq",
        "description": "Supports Hindi, Gujarati, Marathi",
    },
    "IndicConformer (AI4Bharat)": {
        "repo": "ai4bharat/indic-conformer-600m-multilingual",
        "model_type": "ctc_rnnt",
        "description": "Supports 22 Indian languages",
        "trust_remote_code": True,
    },
    "MMS (Facebook)": {
        "repo": "facebook/mms-1b-all",
        "model_type": "ctc",
        "description": "Supports over 1,400 languages (fine-tuning recommended)",
    },
}

# Load model and processor
def load_model_and_processor(model_name):
    config = MODEL_CONFIGS[model_name]
    repo = config["repo"]
    model_type = config["model_type"]
    trust_remote_code = config.get("trust_remote_code", False)

    try:
        if model_name == "IndicConformer (AI4Bharat)":
            model = AutoModel.from_pretrained(repo, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        elif model_name == "MMS (Facebook)":
            model = AutoModelForCTC.from_pretrained(repo)
            processor = AutoProcessor.from_pretrained(repo)
        else:  # AudioX-North
            processor = AutoProcessor.from_pretrained(repo, trust_remote_code=trust_remote_code)
            if model_type == "seq2seq":
                model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, trust_remote_code=trust_remote_code)
            else:
                model = AutoModelForCTC.from_pretrained(repo, trust_remote_code=trust_remote_code)

        return model, processor, model_type
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Compute metrics (WER, CER, RTF)
def compute_metrics(reference, hypothesis, audio_duration, total_time):
    if not reference or not hypothesis:
        return None, None, None, None
    try:
        reference = reference.strip().lower()
        hypothesis = hypothesis.strip().lower()
        wer_score = wer(reference, hypothesis)
        cer_score = cer(reference, hypothesis)
        rtf = total_time / audio_duration if audio_duration > 0 else None
        return wer_score, cer_score, rtf, total_time
    except Exception:
        return None, None, None, None

# Main transcription function
def transcribe_audio(audio_file, selected_models, reference_text=""):
    if not audio_file:
        return "Please upload an audio file.", []
    
    if not selected_models:
        return "Please select at least one model.", []

    table_data = []
    try:
        # Load and preprocess audio once
        audio, sr = librosa.load(audio_file, sr=16000)
        audio_duration = len(audio) / sr

        for model_name in selected_models:
            model, processor, model_type = load_model_and_processor(model_name)
            if isinstance(model_type, str) and model_type.startswith("Error"):
                table_data.append([
                    model_name,
                    f"Error: {model_type}",
                    "-",
                    "-",
                    "-",
                    "-"
                ])
                continue

            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

            start_time = time.time()
            with torch.no_grad():
                if model_type == "seq2seq":
                    input_features = inputs["input_features"]
                    outputs = model.generate(input_features)
                    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                else:  # CTC or RNNT
                    input_values = inputs["input_values"]
                    logits = model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            total_time = time.time() - start_time

            # Compute metrics
            wer_score, cer_score, rtf = "-", "-", "-"
            if reference_text and transcription:
                wer_val, cer_val, rtf_val, _ = compute_metrics(
                    reference_text, transcription, audio_duration, total_time
                )
                wer_score = f"{wer_val:.3f}" if wer_val is not None else "-"
                cer_score = f"{cer_val:.3f}" if cer_val is not None else "-"
                rtf = f"{rtf_val:.3f}" if rtf_val is not None else "-"

            # Add row to table
            table_data.append([
                model_name,
                transcription,
                wer_score,
                cer_score,
                rtf,
                f"{total_time:.2f}s"
            ])

        # Create summary text
        summary = f"**Audio Duration:** {audio_duration:.2f}s\n"
        summary += f"**Models Tested:** {len(selected_models)}\n"
        if reference_text:
            summary += f"**Reference Text:** {reference_text[:100]}{'...' if len(reference_text) > 100 else ''}\n"
        
        return summary, table_data
    except Exception as e:
        return f"Error during transcription: {str(e)}", []

# Create Gradio interface with blocks for better control
def create_interface():
    model_choices = list(MODEL_CONFIGS.keys())
    
    with gr.Blocks(title="Multilingual Speech-to-Text Benchmark") as iface:
        gr.Markdown("""
        # Multilingual Speech-to-Text Benchmark
        Upload an audio file, select one or more models, and optionally provide reference text. 
        The app benchmarks WER, CER, RTF, and Time Taken for each model.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio File (16kHz recommended)", 
                    type="filepath"
                )
                model_selection = gr.CheckboxGroup(
                    choices=model_choices,
                    label="Select Models",
                    value=[model_choices[0]],  # Default to first model
                    interactive=True
                )
                reference_input = gr.Textbox(
                    label="Reference Text (Optional for WER/CER)",
                    placeholder="Enter or paste ground truth text here",
                    lines=8,
                    interactive=True,
                    max_lines=20
                )
                submit_btn = gr.Button("Transcribe", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                summary_output = gr.Markdown(label="Summary", value="Upload an audio file and select models to begin...")
                
                results_table = gr.Dataframe(
                    headers=["Model", "Transcription", "WER", "CER", "RTF", "Time Taken"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    label="Results Comparison",
                    interactive=False,
                    wrap=True,
                    column_widths=[150, 400, 80, 80, 80, 100]
                )
        
        # Connect the function
        submit_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, model_selection, reference_input],
            outputs=[summary_output, results_table]
        )
        
        # Also allow triggering on Enter in reference text
        reference_input.submit(
            fn=transcribe_audio,
            inputs=[audio_input, model_selection, reference_input],
            outputs=[summary_output, results_table]
        )
    
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )