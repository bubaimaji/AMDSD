import torch
import os
import torchaudio
import re
import numpy as np
import pandas as pd
from transformers import AutoProcessor, Wav2Vec2ForCTC
from jiwer import wer

# Load MMS model and processor with language setting
model_id = "facebook/mms-1b-all"
target_lang = "eng"
processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_npy = '/home/suhita/Documents/multimodal/daic/processed_npy/train_combined_df.npy'
#val_npy = '/home/suhita/Documents/multimodal/daic/processed_npy/dev_combined_df.npy'
#val_npy="/home/suhita/Documents/multimodal/daic/processed_npy/test_combined_df1.npy"

# Function to downsample audio samples to 16 kHz
def downsample_audio(audio_array, source_sr, target_sr=16000):
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    resampled_waveform = torchaudio.transforms.Resample(source_sr, target_sr)(waveform)
    resampled_audio = resampled_waveform.squeeze(0).numpy()  # Remove channel dimension
    
    # Ensure the audio length is sufficient for the model
    min_length = target_sr * 1  # 1 second minimum length, adjust as necessary
    if len(resampled_audio) < min_length:
        pad_length = min_length - len(resampled_audio)
        resampled_audio = np.pad(resampled_audio, (0, pad_length), mode='constant')
    
    return resampled_audio

# Function to normalize text
def normalize_text(text):
    text = text.lower()
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text if text else "empty"

# Function to evaluate WER and save ASR transcripts in the data
def evaluate_and_save_asr_transcripts(npy_file, processor, model):
    data = np.load(npy_file, allow_pickle=True).item()
    references = []
    hypotheses = []

    for audio_path, ground_truth_transcript in zip(data['audio_file'], data['transcript']):
        print(f"Processing {audio_path}")
        references.append(normalize_text(ground_truth_transcript))

        audio_input, sampling_rate = torchaudio.load(audio_path)
        audio = audio_input.squeeze().numpy()
        if sampling_rate != 16000:
            audio = downsample_audio(audio, sampling_rate)
        
        # Ensure the audio length is sufficient for the model
        min_length = 16000  # 1 second minimum length at 16 kHz
        if len(audio) < min_length:
            pad_length = min_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
            print(f"Padded {audio_path} to length: {len(audio)}")
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(model.device)
        #print(f"Input tensor shape: {inputs.shape}")
        
        with torch.no_grad():
            logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        hypotheses.append(normalize_text(transcription))

        # Adding transcript to data for saving
        if 'asr_transcript_mms' not in data:
            data['asr_transcript_mms'] = []
        data['asr_transcript_mms'].append(normalize_text(transcription))

    # Calculate WER
    wer_score = wer(references, hypotheses)
    print(f"WER: {wer_score:.4f}")

    np.save(npy_file, data)  # Save the updated data dictionary back to the .npy file
    return wer_score

# Evaluate and save ASR transcripts for train and validation datasets
train_wer_score = evaluate_and_save_asr_transcripts(train_npy, processor, model)
#val_wer_score = evaluate_and_save_asr_transcripts(val_npy, processor, model)

print("Train WER Score:", train_wer_score)
#print("Validation WER Score:", val_wer_score)
