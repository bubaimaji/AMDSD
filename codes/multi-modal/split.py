import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# Define directories for the processed audio files and the new segmented audio files
segmented_audio_dir = "/home/suhita/Documents/multimodal/daic/segmented_audio"
os.makedirs(segmented_audio_dir, exist_ok=True)

# Load the combined datasets from NPY files
train_data = np.load("/home/suhita/Documents/multimodal/daic/processed_npy/train_combined_asr.npy", allow_pickle=True).item()
dev_data = np.load("/home/suhita/Documents/multimodal/daic/processed_npy/dev_combined_asr.npy", allow_pickle=True).item()

train_combined_df = pd.DataFrame(train_data)
dev_combined_df = pd.DataFrame(dev_data)


# Function to split audio into 3-second segments with 50% overlap
def split_audio_segments(audio, sr, segment_duration=3, overlap=0.5):
    segment_length = int(segment_duration * sr)
    overlap_length = int(segment_length * overlap)
    
    segments = []
    for start in range(0, len(audio), segment_length - overlap_length):
        end = start + segment_length
        if end > len(audio):
            break
        segment = audio[start:end]
        segments.append(segment)
    
    return segments, sr

# Function to segment transcripts based on audio segments
def segment_texts(transcript, segment_times, sr):
    words = transcript.split()
    total_duration = segment_times[-1][1]  # End time of the last segment
    word_times = np.linspace(0, total_duration, len(words) + 1)
    
    segments = []
    for start, end in segment_times:
        segment_words = [words[i] for i in range(len(words)) if word_times[i] >= start and word_times[i] < end]
        segments.append(" ".join(segment_words))
    
    return segments

# Function to segment audio files and create new combined datasets
def create_segmented_dataset(combined_df, output_audio_dir, output_csv_file, output_npy_file):
    segmented_data = []
    
    for participant_id in combined_df['audio_file'].apply(lambda x: os.path.basename(x).split('_')[0]).unique():
        participant_df = combined_df[combined_df['audio_file'].str.contains(f"{participant_id}_")]

        full_audio = np.array([])
        full_transcript = ""
        full_asr_mms = ""
        full_asr_whisper = ""
        label = None
        
        for idx, row in participant_df.iterrows():
            audio_file = row['audio_file']
            transcript = row['transcript']
            asr_transcript_mms = row['asr_transcript_mms']
            asr_transcript_whisper = row['asr_transcript_openai/whisper-large']
            label = row['label']

            audio, sr = librosa.load(audio_file, sr=16000)
            full_audio = np.concatenate((full_audio, audio))
            full_transcript += " " + transcript
            full_asr_mms += " " + asr_transcript_mms
            full_asr_whisper += " " + asr_transcript_whisper

        audio_segments, sr = split_audio_segments(full_audio, sr)
        segment_times = [(i * 1.5, (i * 1.5) + 3) for i in range(len(audio_segments))]

        transcript_segments = segment_texts(full_transcript, segment_times, sr)
        asr_mms_segments = segment_texts(full_asr_mms, segment_times, sr)
        asr_whisper_segments = segment_texts(full_asr_whisper, segment_times, sr)

        for i, audio_segment in enumerate(audio_segments):
            segment_path = os.path.join(output_audio_dir, f"{participant_id}_segment_{i}.wav")
            sf.write(segment_path, audio_segment, sr)
            segmented_data.append({
                'audio_file': segment_path,
                'transcript': transcript_segments[i],
                'label': label,
                'asr_transcript_mms': asr_mms_segments[i],
                'asr_transcript_openai/whisper-large': asr_whisper_segments[i]
            })

    segmented_df = pd.DataFrame(segmented_data)
    segmented_df.to_csv(output_csv_file, index=False)
    np.save(output_npy_file, segmented_df.to_dict('list'))
    
    return segmented_df

# Create directories for segmented audio files
os.makedirs(os.path.join(segmented_audio_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(segmented_audio_dir, 'dev'), exist_ok=True)

# Create segmented datasets for train, dev, and test splits
train_segmented_df = create_segmented_dataset(
    train_combined_df,
    os.path.join(segmented_audio_dir, 'train'),
    "/home/suhita/Documents/multimodal/daic/segmented_audio/train_segmented_df.csv",
    "/home/suhita/Documents/multimodal/daic/segmented_audio/train_segmented_df.npy"
)

dev_segmented_df = create_segmented_dataset(
    dev_combined_df,
    os.path.join(segmented_audio_dir, 'dev'),
    "/home/suhita/Documents/multimodal/daic/segmented_audio/dev_segmented_df.csv",
    "/home/suhita/Documents/multimodal/daic/segmented_audio/dev_segmented_df.npy"
)

