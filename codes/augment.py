import pandas as pd
import numpy as np
import random
import re

# Load the pre-processed data from NPY files
data = np.load("/home/suhita/Documents/multimodal/daic/processed_npy/npy/dev_combined_df.npy", allow_pickle=True).item()

# Convert the dictionary back to DataFrame
combined_df = pd.DataFrame(data)
# Function to extract patient ID from the file path
def extract_patient_id(file_path):
    match = re.search(r'/(\d+)_segment', file_path)
    return int(match.group(1)) if match else None

# Apply the function to create a new column for patient ID
combined_df['Participant_ID'] = combined_df['audio_file'].apply(extract_patient_id)

# Function to group responses
def group_responses(df, group_size=10):
    grouped_data = []
    for participant_id in df['Participant_ID'].unique():
        participant_data = df[df['Participant_ID'] == participant_id]
        for i in range(0, len(participant_data), group_size):
            group = participant_data.iloc[i:i + group_size]
            if len(group) == group_size:
                grouped_data.append({
                    'group_id': f"{participant_id}_{i // group_size}",
                    'audio_file': list(group['audio_file']),
                    'transcript': list(group['transcript']),  # Assuming 'ground_truth_transcript' column exists
                    'asr_transcript_mms': list(group['asr_transcript_mms']),  # Assuming 'asr_transcript' column exists
                    'asr_transcript_openai/whisper-large': list(group['asr_transcript_openai/whisper-large']),
                    'label': group['label'].iloc[0]
                })
    return pd.DataFrame(grouped_data)

# Group the responses
grouped_df = group_responses(combined_df)

# Separate groups into depressed and non-depressed
depressed_groups = grouped_df[grouped_df['label'] == 1]
non_depressed_groups = grouped_df[grouped_df['label'] == 0]

# Perform random resampling for depressed groups
resampled_depressed_groups = depressed_groups.sample(n=len(non_depressed_groups), replace=True, random_state=42)

# Combine the resampled depressed groups with non-depressed groups
balanced_df = pd.concat([resampled_depressed_groups, non_depressed_groups]).reset_index(drop=True)

# Shuffle the combined dataframe to ensure randomness
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to CSV
#balanced_df.to_csv("/home/suhita/Documents/multimodal/daic/processed_csv/balanced_combined_df.csv", index=False)

# Optionally save to NPY
np.save("/home/suhita/Documents/multimodal/daic/processed_npy/balanced_combined_df.npy", balanced_df.to_dict('list'))
