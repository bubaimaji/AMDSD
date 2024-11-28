import pandas as pd
import os
import soundfile as sf
import numpy as np
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Define paths to the split CSV files and the directory containing audio and transcript files
train_split_file = "/home/suhita/Documents/multimodal/daic/label/train_split_Depression_AVEC2017.csv"
dev_split_file = "/home/suhita/Documents/multimodal/daic/label/dev_split_Depression_AVEC2017.csv"
test_split_file = "/home/suhita/Documents/multimodal/daic/label/full_test_split - Copy.csv"
transcripts_dir = "/home/suhita/Documents/multimodal/daic/300-492_transcripts"
audio_dir = "/home/suhita/Documents/multimodal/daic/300-492_Audio"

# Load the split data files
train_split_df = pd.read_csv(train_split_file)
dev_split_df = pd.read_csv(dev_split_file)
test_split_df = pd.read_csv(test_split_file)

# Ensure the label column is of integer type immediately after loading
train_split_df['PHQ8_Binary'] = train_split_df['PHQ8_Binary'].astype(int)
dev_split_df['PHQ8_Binary'] = dev_split_df['PHQ8_Binary'].astype(int)
test_split_df['PHQ8_Binary'] = test_split_df['PHQ8_Binary'].astype(int)

# Create directories for saving processed audio files
train_audio_dir = "/home/suhita/Documents/multimodal/daic/audio/train"
dev_audio_dir = "/home/suhita/Documents/multimodal/daic/audio/dev"
test_audio_dir = "/home/suhita/Documents/multimodal/daic/audio/test"
os.makedirs(train_audio_dir, exist_ok=True)
os.makedirs(dev_audio_dir, exist_ok=True)
os.makedirs(test_audio_dir, exist_ok=True)

# Function to normalize text
def normalize_text(text):
    text = text.lower().strip().replace("\n", " ")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to preprocess individual transcript files and extract audio segments
def preprocess_transcript_and_extract_audio(args):
    file_id, transcript_file, audio_file, output_dir = args
    transcript_df = pd.read_csv(transcript_file, sep='\t')
    transcript_df.columns = ['start_time', 'stop_time', 'speaker', 'value']
    transcript_df = transcript_df[transcript_df['speaker'] != 'Ellie']
    transcript_df['value'] = transcript_df['value'].astype(str).apply(normalize_text)
    
    # Load the entire audio file
    audio, sr = sf.read(audio_file)

    segment_data = []
    for idx, row in transcript_df.iterrows():
        start_time = float(row['start_time'])
        stop_time = float(row['stop_time'])
        segment_audio = audio[int(start_time * sr):int(stop_time * sr)]
        segment_path = os.path.join(output_dir, f"{file_id}_segment_{idx}.wav")
        sf.write(segment_path, segment_audio, sr)

        segment_data.append({'segment_path': segment_path, 'transcript': row['value']})

    return segment_data

# Function to create combined dataset with multiprocessing
def create_combined_dataset(split_df, transcripts_dir, audio_dir, output_audio_dir):
    combined_data = []

    # Prepare list of tasks
    tasks = []
    for _, row in split_df.iterrows():
        file_id = int(row['Participant_ID'])
        label = int(row['PHQ8_Binary'])
        audio_file = os.path.join(audio_dir, f"{file_id}_AUDIO.wav")
        transcript_file = os.path.join(transcripts_dir, f"{file_id}_TRANSCRIPT.csv")
        if os.path.exists(audio_file) and os.path.exists(transcript_file):
            tasks.append((file_id, transcript_file, audio_file, output_audio_dir))
        else:
            print(f"Audio or transcript file missing for Participant_ID: {file_id}")

    # Use multiprocessing to process tasks in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = executor.map(preprocess_transcript_and_extract_audio, tasks)
        for result in results:
            combined_data.extend(result)

    combined_df = pd.DataFrame(combined_data)
    return combined_df

# Create combined datasets for train, dev, and test splits
train_combined_df = create_combined_dataset(train_split_df, transcripts_dir, audio_dir, train_audio_dir)
dev_combined_df = create_combined_dataset(dev_split_df, transcripts_dir, audio_dir, dev_audio_dir)
test_combined_df = create_combined_dataset(test_split_df, transcripts_dir, audio_dir, test_audio_dir)

# Save the combined datasets to CSV
train_combined_df.to_csv("/home/suhita/Documents/multimodal/SST/daic_file/train_combined_df.csv", index=False)
dev_combined_df.to_csv("/home/suhita/Documents/multimodal/SST/daic_file/dev_combined_df.csv", index=False)
test_combined_df.to_csv("/home/suhita/Documents/multimodal/SST/daic_file/test_combined_df.csv", index=False)


# Save the combined datasets to NPY
np.save("/home/suhita/Documents/multimodal/SST/daic_file/train_combined_df.npy", train_combined_df.to_dict('list'))
np.save("/home/suhita/Documents/multimodal/SST/daic_file/dev_combined_df.npy", dev_combined_df.to_dict('list'))
np.save("/home/suhita/Documents/multimodal/SST/daic_file/test_combined_df.npy", test_combined_df.to_dict('list'))

# To load the NPY files later
##train_data = np.load("/home/suhita/Documents/multimodal/daic/processed_npy/train_combined_df.npy", allow_pickle=True).item()
#dev_data = np.load("/home/suhita/Documents/multimodal/daic/processed_npy/dev_combined_df.npy", allow_pickle=True).item()
#test_data = np.load("/home/suhita/Documents/multimodal/daic/processed_npy/test_combined_df.npy", allow_pickle=True).item()

# Convert the dictionaries back to DataFrames
#train_combined_df_loaded = pd.DataFrame(train_data)
#dev_combined_df_loaded = pd.DataFrame(dev_data)
#test_combined_df_loaded1 = pd.DataFrame(test_data)
