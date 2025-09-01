import time
import librosa
import gradio as gr
from transformers import AutoModelForCTC, AutoProcessor, pipeline
from jiwer import wer, cer

# ---------------------------
# Load Models (CPU only)
# ---------------------------

# 1. IndicConformer
indic_model_id = "ai4bharat/indic-conformer-600m-multilingual"
indic_processor = AutoProcessor.from_pretrained(indic_model_id)
indic_model = AutoModelForCTC.from_pretrained(indic_model_id)
indic_pipeline = pipeline(
    "automatic-speech-recognition",
    model=indic_model,
    tokenizer=indic_processor.tokenizer,
    feature_extractor=indic_processor.feature_extractor,
    device=-1  # CPU
)

# 2. Facebook MMS (generic multilingual ASR)
mms_model_id = "facebook/mms-1b-all"
mms_processor = AutoProcessor.from_pretrained(mms_model_id)
mms_model = AutoModelForCTC.from_pretrained(mms_model_id)
mms_pipeline = pipeline(
    "automatic-speech-recognition",
    model=mms_model,
    tokenizer=mms_processor.tokenizer,
    feature_extractor=mms_processor.feature_extractor,
    device=-1
)

# 3. Jivi AudioX (North example)
jivi_model_id = "jiviai/audioX-north-v1"
jivi_pipeline = pipeline(
    "automatic-speech-recognition",
    model=jivi_model_id,
    device=-1
)

# ---------------------------
# Utility Functions
# ---------------------------

def evaluate_model(pipeline_fn, audio_path, reference_text):
    # Load audio (resample to 16kHz for consistency)
    speech, sr = librosa.load(audio_path, sr=16000)

    # Measure runtime
    start = time.time()
    result = pipeline_fn(speech)
    end = time.time()

    # Extract transcription
    hypothesis = result["text"]

    # Compute metrics
    word_error = wer(reference_text.lower(), hypothesis.lower())
    char_error = cer(reference_text.lower(), hypothesis.lower())
    rtf = (end - start) / (len(speech) / sr)  # real-time factor

    return hypothesis, word_error, char_error, rtf

def compare_models(audio, reference_text, lang="hi"):
    results = {}

    # IndicConformer
    hyp, w, c, r = evaluate_model(indic_pipeline, audio, reference_text)
    results["IndicConformer"] = (hyp, w, c, r)

    # MMS
    hyp, w, c, r = evaluate_model(mms_pipeline, audio, reference_text)
    results["MMS"] = (hyp, w, c, r)

    # Jivi
    hyp, w, c, r = evaluate_model(jivi_pipeline, audio, reference_text)
    results["Jivi"] = (hyp, w, c, r)

    # Build results table
    table = "| Model | Transcription | WER | CER | RTF |\n"
    table += "|-------|---------------|-----|-----|-----|\n"
    for model, (hyp, w, c, r) in results.items():
        table += f"| {model} | {hyp} | {w:.3f} | {c:.3f} | {r:.3f} |\n"

    return table

# ---------------------------
# Gradio UI
# ---------------------------
demo = gr.Interface(
    fn=compare_models,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio (≤20s recommended)"),
        gr.Textbox(label="Reference Text"),
        gr.Dropdown(choices=["hi", "gu", "ta"], value="hi", label="Language")
    ],
    outputs=gr.Markdown(label="Results"),
    title="ASR Benchmark (CPU mode): IndicConformer vs MMS vs Jivi",
    description="Runs on free CPU Spaces. Upload short audio and reference text. Compares models on WER, CER, and RTF."
)

if __name__ == "__main__":
    demo.launch()