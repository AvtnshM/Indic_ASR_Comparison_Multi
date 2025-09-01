import gradio as gr
import torch
import torchaudio
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModel,
)

# -------------------------------
# Model configurations
# -------------------------------
MODEL_CONFIGS = {
    "Whisper Small (hi)": {
        "repo": "openai/whisper-small",
        "model_type": "seq2seq",
    },
    "IndicConformer 600M": {
        "repo": "ai4bharat/indic-conformer-600m-multilingual",
        "model_type": "ctc",  # but handled specially
        "trust_remote_code": True,
    },
}

# -------------------------------
# Load model and processor
# -------------------------------
def load_model_and_processor(model_name):
    config = MODEL_CONFIGS[model_name]
    repo = config["repo"]
    model_type = config["model_type"]
    trust_remote_code = config.get("trust_remote_code", False)

    try:
        if "indic-conformer" in repo.lower():
            model = AutoModel.from_pretrained(repo, trust_remote_code=True)
            processor = None  # Not required
            return model, processor, model_type
        else:
            processor = AutoProcessor.from_pretrained(repo, trust_remote_code=trust_remote_code)
            if model_type == "seq2seq":
                model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, trust_remote_code=trust_remote_code)
            else:
                model = AutoModelForCTC.from_pretrained(repo, trust_remote_code=trust_remote_code)
            return model, processor, model_type
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# -------------------------------
# Transcription
# -------------------------------
def transcribe_audio(audio_file, model_name, reference_text):
    model, processor, model_type = load_model_and_processor(model_name)
    if model is None:
        return f"⚠️ Failed to load {model_name}: {processor}", ""

    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_file)
    if sampling_rate != 16000:
        speech_array = torchaudio.transforms.Resample(sampling_rate, 16000)(speech_array)
    speech_array = speech_array.squeeze().numpy()

    # Special handling for IndicConformer
    if "indic-conformer" in MODEL_CONFIGS[model_name]["repo"].lower():
        with torch.no_grad():
            transcription = model(torch.tensor(speech_array).unsqueeze(0), "hi", "ctc")
        transcription = transcription[0] if isinstance(transcription, list) else transcription
    else:
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            if model_type == "seq2seq":
                generated_ids = model.generate(inputs["input_features"])
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                logits = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(pred_ids)[0]

    # Compute WER if reference given
    wer_score = None
    if reference_text.strip():
        from jiwer import wer
        wer_score = wer(reference_text, transcription)

    result = f"📝 Transcription: {transcription}"
    if wer_score is not None:
        result += f"\n📊 WER vs reference: {wer_score:.2%}"

    return result, transcription

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Indic ASR Comparison App")

    with gr.Row():
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record Audio")
        model_dropdown = gr.Dropdown(choices=list(MODEL_CONFIGS.keys()), value="Whisper Small (hi)", label="Select Model")

    # ✅ Paste enabled in textbox
    reference_text = gr.Textbox(
        label="Reference Text (optional, paste supported)",
        placeholder="Paste reference transcription here...",
        lines=4,
        interactive=True
    )

    transcribe_btn = gr.Button("Transcribe")
    output_result = gr.Textbox(label="Result", lines=6)
    raw_transcription = gr.Textbox(label="Raw Transcription", lines=4)

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_dropdown, reference_text],
        outputs=[output_result, raw_transcription]
    )

if __name__ == "__main__":
    demo.launch()