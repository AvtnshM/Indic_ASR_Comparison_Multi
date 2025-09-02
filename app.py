import gradio as gr
import torch
import torchaudio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCTC,
    AutoModel,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import librosa
import numpy as np
from jiwer import wer, cer
import time

# Language configurations
LANGUAGE_CONFIGS = {
    "Hindi (हिंदी)": {
        "code": "hi",
        "script": "Devanagari",
        "models": ["AudioX-North", "IndicConformer", "MMS"]
    },
    "Gujarati (ગુજરાતી)": {
        "code": "gu", 
        "script": "Gujarati",
        "models": ["AudioX-North", "IndicConformer", "MMS"]
    },
    "Marathi (मराठी)": {
        "code": "mr",
        "script": "Devanagari", 
        "models": ["AudioX-North", "IndicConformer", "MMS"]
    },
    "Tamil (தமிழ்)": {
        "code": "ta",
        "script": "Tamil",
        "models": ["AudioX-South", "IndicConformer", "MMS"]
    },
    "Telugu (తెలుగు)": {
        "code": "te",
        "script": "Telugu",
        "models": ["AudioX-South", "IndicConformer", "MMS"] 
    },
    "Kannada (ಕನ್ನಡ)": {
        "code": "kn",
        "script": "Kannada",
        "models": ["AudioX-South", "IndicConformer", "MMS"]
    }
}

# Model configurations
MODEL_CONFIGS = {
    "AudioX-North": {
        "repo": "jiviai/audioX-north-v1",
        "model_type": "whisper",
        "description": "Supports Hindi, Gujarati, Marathi",
        "languages": ["hi", "gu", "mr"]
    },
    "AudioX-South": {
        "repo": "jiviai/audioX-south-v1", 
        "model_type": "whisper",
        "description": "Supports Tamil, Telugu, Kannada, Malayalam",
        "languages": ["ta", "te", "kn", "ml"]
    },
    "IndicConformer": {
        "repo": "ai4bharat/indic-conformer-600m-multilingual",
        "model_type": "ctc_rnnt",
        "description": "Supports 22 Indian languages",
        "trust_remote_code": True,
        "languages": ["hi", "gu", "mr", "ta", "te", "kn", "ml", "bn", "pa", "or", "as", "ur"]
    },
    "MMS": {
        "repo": "facebook/mms-1b-all",
        "model_type": "ctc", 
        "description": "Supports 1,400+ languages",
        "languages": ["hi", "gu", "mr", "ta", "te", "kn", "ml"]
    },
}

# Load model and processor
def load_model_and_processor(model_name):
    config = MODEL_CONFIGS[model_name]
    repo = config["repo"]
    model_type = config["model_type"]
    trust_remote_code = config.get("trust_remote_code", False)

    try:
        if model_name == "IndicConformer":
            print(f"Loading {model_name}...")
            try:
                model = AutoModel.from_pretrained(
                    repo, 
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            except Exception as e1:
                print(f"Primary loading failed, trying fallback: {e1}")
                model = AutoModel.from_pretrained(repo, trust_remote_code=True)
            processor = None
            return model, processor, model_type
        
        elif model_name in ["AudioX-North", "AudioX-South"]:
            # Use Whisper processor and model for AudioX variants
            processor = WhisperProcessor.from_pretrained(repo)
            model = WhisperForConditionalGeneration.from_pretrained(repo)
            model.config.forced_decoder_ids = None
            return model, processor, model_type
            
        elif model_name == "MMS":
            model = AutoModelForCTC.from_pretrained(repo)
            processor = AutoProcessor.from_pretrained(repo)
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
def transcribe_audio(audio_file, selected_language, selected_models, reference_text=""):
    if not audio_file:
        return "Please upload an audio file.", [], ""
    
    if not selected_models:
        return "Please select at least one model.", [], ""

    if not selected_language:
        return "Please select a language.", [], ""

    # Get language info
    lang_info = LANGUAGE_CONFIGS[selected_language]
    lang_code = lang_info["code"]
    
    table_data = []
    try:
        # Load and preprocess audio once
        audio, sr = librosa.load(audio_file, sr=16000)
        audio_duration = len(audio) / sr

        for model_name in selected_models:
            # Check if model supports the selected language
            if model_name.replace("AudioX-", "AudioX-") not in lang_info["models"]:
                table_data.append([
                    model_name,
                    f"Language {selected_language} not supported by this model",
                    "-", "-", "-", "-"
                ])
                continue

            model, processor, model_type = load_model_and_processor(model_name)
            if isinstance(model_type, str) and model_type.startswith("Error"):
                table_data.append([
                    model_name,
                    f"Error: {model_type}",
                    "-", "-", "-", "-"
                ])
                continue

            start_time = time.time()
            
            try:
                if model_name == "IndicConformer":
                    # AI4Bharat specific processing
                    wav = torch.from_numpy(audio).unsqueeze(0)
                    if torch.max(torch.abs(wav)) > 0:
                        wav = wav / torch.max(torch.abs(wav))
                    
                    with torch.no_grad():
                        transcription = model(wav, lang_code, "rnnt")
                        if isinstance(transcription, list):
                            transcription = transcription[0] if transcription else ""
                        transcription = str(transcription).strip()
                
                elif model_name in ["AudioX-North", "AudioX-South"]:
                    # AudioX Whisper-based processing
                    if sr != 16000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    
                    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
                    
                    with torch.no_grad():
                        predicted_ids = model.generate(
                            input_features, 
                            task="transcribe", 
                            language=lang_code
                        )
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                else:  # MMS
                    # Standard CTC processing for MMS
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                    
                    with torch.no_grad():
                        input_values = inputs["input_values"]
                        logits = model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            except Exception as e:
                transcription = f"Processing error: {str(e)}"

            total_time = time.time() - start_time

            # Compute metrics
            wer_score, cer_score, rtf = "-", "-", "-"
            if reference_text and transcription and not transcription.startswith("Processing error"):
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
        summary = f"**Language:** {selected_language} ({lang_code})\n"
        summary += f"**Audio Duration:** {audio_duration:.2f}s\n"
        summary += f"**Models Tested:** {len(selected_models)}\n"
        if reference_text:
            summary += f"**Reference Text:** {reference_text[:100]}{'...' if len(reference_text) > 100 else ''}\n"
        
        # Create copyable text output
        copyable_text = "MULTILINGUAL SPEECH-TO-TEXT BENCHMARK RESULTS\n" + "="*55 + "\n\n"
        copyable_text += f"Language: {selected_language} ({lang_code})\n"
        copyable_text += f"Script: {lang_info['script']}\n"
        copyable_text += f"Audio Duration: {audio_duration:.2f}s\n"
        copyable_text += f"Models Tested: {len(selected_models)}\n"
        if reference_text:
            copyable_text += f"Reference Text: {reference_text}\n"
        copyable_text += "\n" + "-"*55 + "\n\n"
        
        for i, row in enumerate(table_data):
            copyable_text += f"MODEL {i+1}: {row[0]}\n"
            copyable_text += f"Transcription: {row[1]}\n"
            copyable_text += f"WER: {row[2]}\n"
            copyable_text += f"CER: {row[3]}\n"
            copyable_text += f"RTF: {row[4]}\n"
            copyable_text += f"Time Taken: {row[5]}\n"
            copyable_text += "\n" + "-"*35 + "\n\n"
        
        return summary, table_data, copyable_text
    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}"
        return error_msg, [], error_msg

# Create Gradio interface
def create_interface():
    language_choices = list(LANGUAGE_CONFIGS.keys())
    
    with gr.Blocks(title="Multilingual Speech-to-Text Benchmark", css="""
        .language-info { background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .copy-area { font-family: monospace; font-size: 12px; }
    """) as iface:
        gr.Markdown("""
        # 🌏 Multilingual Speech-to-Text Benchmark
        
        Compare ASR models across **6 Indian Languages** with comprehensive metrics.
        
        **Supported Languages:** Hindi, Gujarati, Marathi, Tamil, Telugu, Kannada
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Language selection
                language_selection = gr.Dropdown(
                    choices=language_choices,
                    label="🗣️ Select Language",
                    value=language_choices[0],
                    interactive=True
                )
                
                audio_input = gr.Audio(
                    label="📹 Upload Audio File (16kHz recommended)", 
                    type="filepath"
                )
                
                # Dynamic model selection based on language
                model_selection = gr.CheckboxGroup(
                    choices=["AudioX-North", "IndicConformer", "MMS"],
                    label="🤖 Select Models",
                    value=["AudioX-North", "IndicConformer"],
                    interactive=True
                )
                
                reference_input = gr.Textbox(
                    label="📝 Reference Text (optional, paste supported)",
                    placeholder="Paste reference transcription here...",
                    lines=4,
                    interactive=True
                )
                
                submit_btn = gr.Button("🚀 Run Multilingual Benchmark", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    label="📊 Summary", 
                    value="Select language, upload audio file and choose models to begin..."
                )
                
                results_table = gr.Dataframe(
                    headers=["Model", "Transcription", "WER", "CER", "RTF", "Time"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    label="🏆 Results Comparison",
                    interactive=False,
                    wrap=True,
                    column_widths=[120, 350, 60, 60, 60, 80]
                )
                
                # Copyable results section
                with gr.Group():
                    gr.Markdown("### 📋 Export Results")
                    copyable_output = gr.Textbox(
                        label="Copy-Paste Friendly Results",
                        lines=12,
                        max_lines=25,
                        show_copy_button=True,
                        interactive=False,
                        elem_classes="copy-area",
                        placeholder="Benchmark results will appear here..."
                    )
        
        # Update model choices based on language selection
        def update_model_choices(selected_language):
            if not selected_language:
                return gr.CheckboxGroup(choices=[], value=[])
            
            lang_info = LANGUAGE_CONFIGS[selected_language]
            available_models = lang_info["models"]
            
            # Map display names
            model_map = {
                "AudioX-North": "AudioX-North", 
                "AudioX-South": "AudioX-South",
                "IndicConformer": "IndicConformer", 
                "MMS": "MMS"
            }
            
            available_choices = [model_map[model] for model in available_models if model in model_map]
            default_selection = available_choices[:2] if len(available_choices) >= 2 else available_choices
            
            return gr.CheckboxGroup(choices=available_choices, value=default_selection)
        
        # Connect language selection to model updates
        language_selection.change(
            fn=update_model_choices,
            inputs=[language_selection],
            outputs=[model_selection]
        )
        
        # Connect the main function
        submit_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language_selection, model_selection, reference_input],
            outputs=[summary_output, results_table, copyable_output]
        )
        
        reference_input.submit(
            fn=transcribe_audio,
            inputs=[audio_input, language_selection, model_selection, reference_input],
            outputs=[summary_output, results_table, copyable_output]
        )
        
        # Language information display
        gr.Markdown("""
        ---
        ### 🔤 Language & Model Support Matrix
        
        | Language | Script | AudioX-North | AudioX-South | IndicConformer | MMS |
        |----------|---------|-------------|-------------|---------------|-----|
        | Hindi | Devanagari | ✅ | ❌ | ✅ | ✅ |
        | Gujarati | Gujarati | ✅ | ❌ | ✅ | ✅ |
        | Marathi | Devanagari | ✅ | ❌ | ✅ | ✅ |
        | Tamil | Tamil | ❌ | ✅ | ✅ | ✅ |
        | Telugu | Telugu | ❌ | ✅ | ✅ | ✅ |
        | Kannada | Kannada | ❌ | ✅ | ✅ | ✅ |
        
        ### 💡 Tips:
        - **Models auto-filter** based on selected language
        - **Reference Text**: Enable WER/CER calculation by providing ground truth
        - **Copy Results**: Export formatted results using the copy button
        - **Best Performance**: Use AudioX models for their specialized languages
        """)
    
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