import time
import os
import evaluate
from datasets import load_dataset
from huggingface_hub import login
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# 🔑 Authenticate using HF_TOKEN secret
login(token=os.environ.get("HF_TOKEN"))

# -----------------
# Load evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# -----------------
# Small sample dataset for Hindi
# (free Spaces can't handle large test sets)
test_ds = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test[:3]")

# Extract references + audio
refs = [x["sentence"] for x in test_ds]
audio_data = [x["audio"]["array"] for x in test_ds]

results = {}

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
            "Transcriptions": preds,
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
# Run evaluations
for label, cfg in models.items():
    print(f"Running {label}...")
    results[label] = evaluate_model(cfg["name"], cfg["pipeline_kwargs"])

print(results)