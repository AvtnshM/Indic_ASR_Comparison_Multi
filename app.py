import os
import time
import evaluate
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoProcessor, AutoModelForCTC

# Get HF token from secret (for gated repos like Jivi)
hf_token = os.getenv("HF_TOKEN")

# Load Hindi dataset (tiny sample for speed)
test_ds = load_dataset("mozilla-foundation/common_voice_11_0_hi", split="test[:3]")

# Metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Models to compare
models = {
    "IndicConformer (AI4Bharat)": "ai4bharat/IndicConformer-hi",
    "AudioX-North (Jivi AI)": "jiviai/audioX-north-v1",
    "MMS (Facebook)": "facebook/mms-1b-all"
}

results = []

for model_name, model_id in models.items():
    print(f"\n🔹 Running {model_name} ...")
    try:
        # Init pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            tokenizer=model_id,
            feature_extractor=model_id,
            use_auth_token=hf_token if "jiviai" in model_id else None
        )

        # Test loop
        for sample in test_ds:
            audio = sample["audio"]["array"]
            ref_text = sample["sentence"]

            start_time = time.time()
            pred_text = asr(audio)["text"]
            elapsed = time.time() - start_time

            # Metrics
            wer = wer_metric.compute(predictions=[pred_text], references=[ref_text])
            cer = cer_metric.compute(predictions=[pred_text], references=[ref_text])
            rtf = elapsed / (len(audio) / 16000)  # real-time factor (audio length at 16kHz)

            results.append({
                "Model": model_name,
                "Reference": ref_text,
                "Prediction": pred_text,
                "WER": round(wer, 3),
                "CER": round(cer, 3),
                "RTF": round(rtf, 3)
            })

    except Exception as e:
        results.append({
            "Model": model_name,
            "Reference": "-",
            "Prediction": "-",
            "WER": None,
            "CER": None,
            "RTF": None,
            "Error": str(e)
        })

# Convert results to DataFrame
df = pd.DataFrame(results)
print("\n===== Final Comparison =====")
print(df.to_string(index=False))