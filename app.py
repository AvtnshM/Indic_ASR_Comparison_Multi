import time
import os
import evaluate
import gradio as gr
from datasets import load_dataset
from transformers import pipeline

# -----------------
# Load evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# -----------------
# Small sample dataset for Hindi
test_ds = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test[:3]")

# Extract references + audio
refs = [x["sentence"] for x in test_ds]
audio_data = [x["audio"]["array"] for x in test_ds]

# -----------------
# Helper to evaluate model
def evaluate_model(model_name, pipeline_kwargs=None):
    try:
        start = time.time()
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=-1,  # CPU only
            **(pipeline_kwargs or {})
        )

        preds = []
        for audio in audio_data:
            out = asr_pipeline(audio, chunk_length_s=30, return_timestamps=False)
            preds.append(out["text"])

        end = time.time()
        rtf = (end - start) / sum(len(a) / 16000 for a in audio_data)

        return {
            "WER": wer_metric.compute(predictions=preds, references=refs),
            "CER": cer_metric.compute(predictions=preds, references=refs),
            "RTF": rtf
        }

    except Exception as e:
        return {"Error": str(e)}

# -----------------
# Models to test
models = {
    "IndicConformer (AI4Bharat)": {
        "name": "ai4bharat/IndicConformer-Hi",
        "pipeline_kwargs": {"trust_remote_code": True}
    },
    "AudioX-North (Jivi AI)": {
        "name": "jiviai/audioX-north-v1",
        "pipeline_kwargs": {"use_auth_token": os.environ.get("HF_TOKEN")}
    },
    "MMS (Facebook)": {
        "name": "facebook/mms-1b-all",
        "pipeline_kwargs": {}
    }
}

# -----------------
# Gradio interface
def run_evaluations():
    rows = []
    for label, cfg in models.items():
        res = evaluate_model(cfg["name"], cfg["pipeline_kwargs"])
        if "Error" in res:
            rows.append([label, res["Error"], "-", "-"])
        else:
            rows.append([label, f"{res['WER']:.3f}", f"{res['CER']:.3f}", f"{res['RTF']:.2f}"])
    return rows

with gr.Blocks() as demo:
    gr.Markdown("## ASR Benchmark Comparison (Hindi Sample)\nEvaluating **WER, CER, RTF** across models.")
    btn = gr.Button("Run Evaluation")
    table = gr.Dataframe(headers=["Model", "WER", "CER", "RTF"], datatype=["str", "str", "str", "str"], interactive=False)

    btn.click(fn=run_evaluations, outputs=table)

if __name__ == "__main__":
    demo.launch()