import time
import torch
import gradio as gr
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from jiwer import wer, cer

# -----------------------------
# Load sample dataset (Hindi)
# -----------------------------
# We’ll use a few samples for faster CPU benchmarking
test_ds = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test[:3]")

# -----------------------------
# Model configs
# -----------------------------
models = {
    "IndicWhisper (Hindi)": {
        "id": "ai4bharat/indicwhisper-large-hi",
        "type": "whisper",
    },
    "IndicConformer": {
        "id": "ai4bharat/indic-conformer-600m-multilingual",
        "type": "conformer",
    },
    "MMS (Facebook)": {
        "id": "facebook/mms-1b-all",
        "type": "conformer",
    },
}

# -----------------------------
# Helper function for inference
# -----------------------------
def evaluate_model(name, cfg, dataset):
    print(f"\nRunning {name}...")
    start_time = time.time()

    if cfg["type"] == "whisper":
        processor = WhisperProcessor.from_pretrained(cfg["id"])
        model = WhisperForConditionalGeneration.from_pretrained(cfg["id"]).to("cpu")
        pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=-1)

    else:  # Conformer (Indic or MMS)
        processor = AutoProcessor.from_pretrained(cfg["id"], trust_remote_code=True)
        model = AutoModelForCTC.from_pretrained(cfg["id"], trust_remote_code=True).to("cpu")
        pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=-1)

    preds, refs = [], []
    for sample in dataset:
        audio = sample["audio"]["array"]
        ref_text = sample["sentence"]
        out = pipe(audio)
        preds.append(out["text"])
        refs.append(ref_text)

    elapsed = time.time() - start_time
    rtf = elapsed / sum(len(s["audio"]["array"]) / 16000 for s in dataset)

    return {
        "WER": wer(refs, preds),
        "CER": cer(refs, preds),
        "RTF": rtf,
        "Predictions": preds,
        "References": refs,
    }

# -----------------------------
# Gradio UI
# -----------------------------
def run_comparison():
    results = {}
    for name, cfg in models.items():
        results[name] = evaluate_model(name, cfg, test_ds)
    return results

demo = gr.Interface(
    fn=run_comparison,
    inputs=[],
    outputs="json",
    title="Indic ASR Benchmark (CPU)",
    description="Compares IndicWhisper (Hindi), IndicConformer, and MMS on WER, CER, and RTF.",
)

if __name__ == "__main__":
    demo.launch()