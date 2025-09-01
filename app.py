import time
import torch
import gradio as gr
import torchaudio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoProcessor, AutoModelForCTC, pipeline
)
from jiwer import wer, cer

# Utility to load audio and resample to 16 kHz
def load_audio(fp):
    waveform, sr = torchaudio.load(fp)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.squeeze(0), 16000

# Evaluation function
def eval_model(name, cfg, file, ref):
    waveform, sr = load_audio(file)
    start = time.time()

    if cfg["type"] == "whisper":
        proc = WhisperProcessor.from_pretrained(cfg["id"])
        model = WhisperForConditionalGeneration.from_pretrained(cfg["id"])
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            device=-1
        )
    else:
        proc = AutoProcessor.from_pretrained(cfg["id"], trust_remote_code=True)
        model = AutoModelForCTC.from_pretrained(cfg["id"], trust_remote_code=True)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            device=-1
        )

    result = pipe(waveform)
    hyp = result["text"].lower()
    w = wer(ref.lower() if ref else "", hyp) if ref else None
    c = cer(ref.lower() if ref else "", hyp) if ref else None
    rtf = (time.time() - start) / (waveform.shape[0] / sr)

    return {"Transcription": hyp, "WER": w, "CER": c, "RTF": rtf}

# Model configs
MODELS = {
    "IndicConformer (AI4Bharat)": {"id": "ai4bharat/indic-conformer-600m-multilingual", "type": "conformer"},
    "AudioX-North (Jivi AI)": {"id": "jiviai/audioX-north-v1", "type": "whisper"},
    "MMS (Facebook)": {"id": "facebook/mms-1b-all", "type": "conformer"},
}

# Gradio interface logic
def compare_all(audio, reference, language):
    results = {}
    for name, cfg in MODELS.items():
        try:
            results[name] = eval_model(name, cfg, audio, reference)
        except Exception as e:
            results[name] = {"Error": str(e)}
    return results

demo = gr.Interface(
    fn=compare_all,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio (<=20s recommended)"),
        gr.Textbox(label="Reference Transcript (optional)"),
        gr.Dropdown(choices=["hi","gu","ta"], label="Language", value="hi")
    ],
    outputs=gr.JSON(label="Benchmark Results"),
    title="Indic ASR Benchmark (CPU-only)",
    description="Compare IndicConformer, AudioX-North, and MMS on WER, CER, and RTF."
)

if __name__ == "__main__":
    demo.launch()