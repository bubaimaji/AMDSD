import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
import librosa
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from transformers import WhisperProcessor, WhisperModel

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load .npy data
train_data = np.load('/home/suhita/Documents/multimodal/daic/segmented_audio/train_segmented_df.npy', allow_pickle=True).item()
dev_data = np.load("/home/suhita/Documents/multimodal/daic/segmented_audio/dev_segmented_df.npy", allow_pickle=True).item()

# Convert the dictionaries into pandas DataFrames
train_data_df = pd.DataFrame(train_data)
dev_data_df = pd.DataFrame(dev_data)

# Combine and shuffle the data
combined_data_df = pd.concat([train_data_df, dev_data_df], ignore_index=True)
combined_data_df = shuffle(combined_data_df, random_state=42)

# Load pre-trained Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
audio_model = WhisperModel.from_pretrained("openai/whisper-medium").to(device)

# Freeze the Whisper model parameters
for param in audio_model.parameters():
    param.requires_grad = False

# Load GPT-2 model and tokenizer
gpt2_model = GPT2Model.from_pretrained("gpt2").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Freeze the GPT-2 model parameters
for param in gpt2_model.parameters():
    param.requires_grad = False

# Fully connected layer to map speech embedding to GPT-2 hidden space
fc_layer = nn.Linear(audio_model.config.hidden_size, gpt2_model.config.n_embd).to(device)

# Define a contextual prompt for depression detection
prompt = "Analyze the following transcript to detect signs of depression: "

# Function to extract audio features using Whisper's encoder
def extract_audio_features(audio_file):
    audio_input, _ = librosa.load(audio_file, sr=16000)  # Load audio
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        encoder_outputs = audio_model.encoder(input_features)  # Use only the encoder part
        features = encoder_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return features

# Function to implement in-context fusion with optional prompt
def in_context_fusion(speech_embeddings, transcript, use_prompt=False):
    # Apply the fully connected layer to map speech embedding to GPT-2 hidden space
    speech_context = torch.relu(fc_layer(speech_embeddings)).unsqueeze(1)
    
    # Optionally add prompt to the transcript
    if use_prompt:
        transcript = prompt + transcript
    
    # Tokenize the transcript with or without the prompt
    inputs = gpt2_tokenizer(transcript, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Get the text token embeddings from GPT-2
    text_embeddings = gpt2_model.wte(input_ids)
    
    # Concatenate speech context with text embeddings
    combined_embeddings = torch.cat([speech_context, text_embeddings], dim=1)
    
    # Adjust attention mask to consider speech context
    extended_attention_mask = torch.cat([torch.ones(speech_context.size(0), 1).to(device), attention_mask], dim=1)
    
    # Pass through GPT-2 transformer layers
    with torch.no_grad():  # Ensure no gradients are computed
        outputs = gpt2_model(inputs_embeds=combined_embeddings, attention_mask=extended_attention_mask)
    
    return outputs.last_hidden_state

# Function to process a batch of data and extract embeddings with prompt option
def extract_embeddings(audio_files, transcripts, use_prompt=False):
    all_embeddings = []
    for audio_file, transcript in zip(audio_files, transcripts):
        # Extract speech embeddings using Whisper's encoder
        speech_embeddings = extract_audio_features(audio_file)
        # Perform in-context fusion using the speech embeddings and modified transcript
        last_hidden_state = in_context_fusion(speech_embeddings, transcript, use_prompt)
        # Pooling across the sequence to get a single vector per sample
        pooled_output = last_hidden_state.mean(dim=1)
        # Detach from the computation graph and convert to numpy
        all_embeddings.append(pooled_output.detach().cpu().numpy())
    
    return np.vstack(all_embeddings)

# Prepare data
audio_files = combined_data_df['audio_file'].tolist()
transcripts = combined_data_df['transcript'].tolist()
labels = combined_data_df['label'].values

# Shuffle and split the data
audio_files, transcripts, labels = shuffle(audio_files, transcripts, labels, random_state=42)
train_files, test_files, train_transcripts, test_transcripts, train_labels, test_labels = train_test_split(
    audio_files, transcripts, labels, test_size=0.25, random_state=42
)

# Extract embeddings with prompt
train_embeddings_with_prompt = extract_embeddings(train_files, train_transcripts, use_prompt=True)
test_embeddings_with_prompt = extract_embeddings(test_files, test_transcripts, use_prompt=True)

# Standardize the embeddings
scaler_with_prompt = StandardScaler()
train_embeddings_with_prompt = scaler_with_prompt.fit_transform(train_embeddings_with_prompt)
test_embeddings_with_prompt = scaler_with_prompt.transform(test_embeddings_with_prompt)

# Train and evaluate SVM classifier for with prompt using linear kernel
print("Training with Linear Kernel with Prompt:")
svm_classifier_linear_with_prompt = svm.SVC(kernel='linear', probability=True)
svm_classifier_linear_with_prompt.fit(train_embeddings_with_prompt, train_labels)
test_predictions_linear_with_prompt = svm_classifier_linear_with_prompt.predict(test_embeddings_with_prompt)

print(f"Linear Kernel Accuracy with Prompt: {accuracy_score(test_labels, test_predictions_linear_with_prompt)}")
print("Linear Kernel Classification Report with Prompt:")
print(classification_report(test_labels, test_predictions_linear_with_prompt))

# Calculate and print total trainable parameters excluding Whisper encoder
total_trainable_params = sum(
    p.numel() for model in [gpt2_model, fc_layer] for p in model.parameters() if p.requires_grad
)
print(f"Total trainable parameters excluding Whisper encoder: {total_trainable_params}")
