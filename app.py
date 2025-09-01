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
        return "Please upload an audio file."

    results = []
    try:
        # Load and preprocess audio once
        audio, sr = librosa.load(audio_file, sr=16000)
        audio_duration = len(audio) / sr

        for model_name in selected_models:
            model, processor, model_type = load_model_and_processor(model_name)
            if isinstance(model_type, str) and model_type.startswith("Error"):
                results.append(f"{model_name}: {model_type}")
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
            wer_score, cer_score, rtf, total_time_tracked = "", "", "", ""
            if reference_text and transcription:
                wer_score, cer_score, rtf, total_time_tracked = compute_metrics(
                    reference_text, transcription, audio_duration, total_time
                )
                wer_score = round(wer_score, 3) if wer_score is not None else ""
                cer_score = round(cer_score, 3) if cer_score is not None else ""
                rtf = round(rtf, 3) if rtf is not None else ""
                total_time_tracked = round(total_time_tracked, 2) if total_time_tracked is not None else ""

            result = (
                f"### {model_name}\n"
                f"- **Transcription:** {transcription}\n"
                f"- **WER:** {wer_score}\n"
                f"- **CER:** {cer_score}\n"
                f"- **RTF:** {rtf}\n"
                f"- **Time Taken (s):** {total_time_tracked}\n"
            )
            results.append(result)

        return "\n\n".join(results)
    except Exception as e:
        return f"Error during transcription: {str(e)}"


# Gradio interface
def create_interface():
    model_choices = list(MODEL_CONFIGS.keys())
    return gr.Interface(
        fn=transcribe_audio,
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio File (16kHz recommended)"),
            gr.CheckboxGroup(choices=model_choices, label="Select Models", value=model_choices),
            gr.Textbox(label="Reference Text (Optional for WER/CER)", placeholder="Enter or paste ground truth text here", lines=3),
        ],
        outputs=gr.Markdown(label="Results"),
        title="Multilingual Speech-to-Text Benchmark",
        description="Upload an audio file, select one or more models, and optionally provide reference text. The app benchmarks WER, CER, RTF, and Time Taken for each model.",
        allow_flagging="never",
    )


if __name__ == "__main__":
    iface = create_interface()
    iface.launch()