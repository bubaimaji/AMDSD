import torch
import os
import torchaudio
import re
import numpy as np
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

# Load Whisper models and processors with language setting
model_names = ["openai/whisper-large"]
models = {}
processors = {}
for model_name in model_names:
    processors[model_name] = WhisperProcessor.from_pretrained(model_name, language="en")
    models[model_name] = WhisperForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_npy = '/home/suhita/Documents/multimodal/daic/processed_npy/train_combined_df.npy'
#val_npy = '/home/suhita/Documents/multimodal/daic/processed_npy/dev_combined_df.npy'


# Function to downsample audio samples to 16 kHz
def downsample_audio(audio_array, source_sr, target_sr=16000):
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    resampled_waveform = torchaudio.transforms.Resample(source_sr, target_sr)(waveform)
    resampled_audio = resampled_waveform.squeeze(0).numpy()  # Remove channel dimension
    return resampled_audio

# Function to normalize text
def normalize_text(text):
    text = text.lower()
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text

# Function to evaluate WER and save ASR transcripts in the data
def evaluate_and_save_asr_transcripts(npy_file, processors, models):
    data = np.load(npy_file, allow_pickle=True).item()
    wer_scores = {model_name: [] for model_name in model_names}
    references = []
    hypotheses = {model_name: [] for model_name in model_names}

    for audio_path, ground_truth_transcript in zip(data['audio_file'], data['transcript']):
        print(f"Processing {audio_path}")
        references.append(normalize_text(ground_truth_transcript))

        for model_name in model_names:
            processor = processors[model_name]
            model = models[model_name]
            
            audio_input, sampling_rate = torchaudio.load(audio_path)
            audio = audio_input.squeeze().numpy()
            if sampling_rate != 16000:
                audio = downsample_audio(audio, sampling_rate)
            
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            hypotheses[model_name].append(normalize_text(transcription))
            # Adding transcript to data for saving
            if f'asr_transcript_{model_name}' not in data:
                data[f'asr_transcript_{model_name}'] = []
            data[f'asr_transcript_{model_name}'].append(normalize_text(transcription))

    for model_name in model_names:
        # Ensure all entries are added even if previously missing
        if f'asr_transcript_{model_name}' not in data:
            data[f'asr_transcript_{model_name}'] = hypotheses[model_name]
        wer_score = wer(references, hypotheses[model_name])
        wer_scores[model_name].append(wer_score)
        print(f"Model: {model_name}, WER: {wer_score:.4f}")

    np.save(npy_file, data)  # Save the updated data dictionary back to the .npy file
    return wer_scores

# Evaluate and save ASR transcripts for train and validation datasets
train_wer_scores = evaluate_and_save_asr_transcripts(train_npy, processors, models)
#val_wer_scores = evaluate_and_save_asr_transcripts(val_npy, processors, models)

print("Train WER Scores:", train_wer_scores)
#print("Validation WER Scores:", val_wer_scores)

# Save the WER results to a CSV file
train_wer_df = pd.DataFrame(train_wer_scores)
#val_wer_df = pd.DataFrame(val_wer_scores)
train_wer_df.to_csv("train_wer_results.csv", index=False)
#val_wer_df.to_csv("val_wer_results.csv", index=False)