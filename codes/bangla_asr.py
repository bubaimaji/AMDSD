import os
import librosa
import torch
import torchaudio
import numpy as np
import pandas as pd
from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
npy_file = "/home/suhita/Documents/multimodal/bangla/df_speaker-wise.npy"
model_path = "bangla-speech-processing/BanglaASR"

# Load Whisper models and processors with language setting
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

# Function to downsample audio samples to 16 kHz
def downsample_audio(audio_array, source_sr, target_sr=16000):
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    resampled_waveform = torchaudio.transforms.Resample(source_sr, target_sr)(waveform)
    resampled_audio = resampled_waveform.squeeze(0).numpy()  # Remove channel dimension
    return resampled_audio

# Function to process audio and generate ASR transcripts
def process_audio_and_generate_asr(npy_file, processor, model):
    data = np.load(npy_file, allow_pickle=True).item()
    if 'asr_transcript' not in data:
        data['asr_transcript'] = []

    for audio_path in data['path']:
        print(f"Processing {audio_path}")

        audio_input, sampling_rate = torchaudio.load(audio_path)
        audio = audio_input.squeeze().numpy()
        if sampling_rate != 16000:
            audio = downsample_audio(audio, sampling_rate)

        input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(inputs=input_features)[0]
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        data['asr_transcript'].append(transcription)
        print(f"ASR Result for {audio_path}: {transcription}")

    np.save(npy_file, data)  # Save the updated data dictionary back to the .npy file

# Process audio and generate ASR transcripts for the Bengali dataset
process_audio_and_generate_asr(npy_file, processor, model)

# Optionally save the processed data to CSV
data = np.load(npy_file, allow_pickle=True).item()
df = pd.DataFrame(data)
df.to_csv("/home/suhita/Documents/multimodal/bangla/bangla_processed_data_with_asr.csv", index=False)

print("ASR processing and saving completed.")
