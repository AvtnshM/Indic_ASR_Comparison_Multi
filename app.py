import os
import time
import torch
import gradio as gr
from datasets import load_dataset
from transformers import pipeline
import evaluate

# Get token for gated repos
hf_token = os.getenv("HF_TOKEN")

# Load Hindi audio samples (Common Voice Hindi test subset)
test_ds = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "hi",
    split="test[:3]",
    use_auth_token=hf_token,
)

# Prepare references and audio arrays
refs = [sample["sentence"] for sample in test_ds]
audio_samples = [sample["audio"]["array"] for sample in test_ds]

# Metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Models to test
MODELS = {
    "IndicConformer (AI4Bharat)": {
        "model_id": "ai4bharat/indic-conformer-600m-multilingual",
        "trust_remote_code": True,
        "auth": None
    },
    "AudioX-North (Jivi AI)": {
        "model_id": "jiviai/audioX-north-v1",
        "trust_remote_code": False,
        "auth": hf_token
    },
    "MMS (Facebook)": {
        "model_id": "facebook/mms-1b-all",
        "trust_remote_code": False,
        "auth": None
    }
}

def eval_model(model_info):
    args = {
        "model": model_info["model_id"],
        "device": -1  # CPU only
    }
    if model_info["trust_remote_code"]:
        args["trust_remote_code"] = True
    if model_info["auth"]:
        args["use_auth_token"] = model_info["auth"]

    asr = pipeline("automatic-speech-recognition", **args)
    preds = []
    start = time.time()
    for audio in audio_samples:
        out = asr(audio)
        preds.append(out["text"].strip())
    elapsed = time.time() - start

    total_len = sum(len(a) for a in audio_samples) / 16000
    rtf = elapsed / total_len

    return {
        "WER": wer_metric.compute(predictions=preds, references=refs),
        "CER": cer_metric.compute(predictions=preds, references=refs),
        "RTF": rtf
    }

def run_all():
    rows = []
    for name, cfg in MODELS.items():
        try:
            res = eval_model(cfg)
            rows.append([name, f"{res['WER']:.3f}", f"{res['CER']:.3f}", f"{res['RTF']:.2f}"])
        except Exception as e:
            rows.append([name, "Error", "Error", "Error"])
    return rows

with gr.Blocks() as demo:
    gr.Markdown("### ASR Model Benchmark (Hindi Samples)\nWER, CER, and RTF comparison.")
    btn = gr.Button("Run Benchmark")
    table = gr.Dataframe(
        headers=["Model", "WER", "CER", "RTF"],
        datatype=["str", "str", "str", "str"],
        interactive=False
    )
    btn.click(run_all, outputs=table)

if __name__ == "__main__":
    demo.launch()