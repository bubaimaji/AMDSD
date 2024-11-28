import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the .npy data
train_data = np.load('/home/suhita/Documents/multimodal/daic/segmented_audio/balanced_train_combined_df.npy', allow_pickle=True).item()
dev_data = np.load("/home/suhita/Documents/multimodal/daic/segmented_audio/dev_segmented_df.npy", allow_pickle=True).item()

# Convert the dictionaries into pandas DataFrames
train_data_df = pd.DataFrame(train_data)
dev_data_df = pd.DataFrame(dev_data)

# Combine and shuffle the data
combined_data_df = pd.concat([train_data_df, dev_data_df], ignore_index=True)
combined_data_df = shuffle(combined_data_df, random_state=42)

# Load Falcon-7B-Instruct model and tokenizer with 4-bit precision using bitsandbytes
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
falcon_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", quantization_config=bnb_config).to(device)
falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

# Freeze all model parameters
for param in falcon_model.parameters():
    param.requires_grad = False

# Function to implement in-context fusion (text only)
def in_context_fusion(transcript):
    # Tokenize the transcript
    inputs = falcon_tokenizer(transcript, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Get the text embeddings from Falcon-7B-Instruct
    outputs = falcon_model(input_ids=input_ids, attention_mask=attention_mask)

    # Return the last hidden state
    return outputs.last_hidden_state

# Function to process a batch of data and extract embeddings
def extract_embeddings(transcripts, batch_size=4):
    all_embeddings = []
    for i in range(0, len(transcripts), batch_size):
        batch_transcripts = transcripts[i:i+batch_size]
        batch_embeddings = []
        for transcript in batch_transcripts:
            last_hidden_state = in_context_fusion(transcript)
            pooled_output = last_hidden_state.mean(dim=1)  # Pooling across the sequence
            batch_embeddings.append(pooled_output.detach().cpu().numpy())  # Detach before converting to numpy
        all_embeddings.append(np.vstack(batch_embeddings))

    return np.vstack(all_embeddings)

# Prepare data
transcripts = combined_data_df['transcript'].tolist()
labels = combined_data_df['label'].values

# Shuffle and split the data
transcripts, labels = shuffle(transcripts, labels, random_state=42)
train_transcripts, test_transcripts, train_labels, test_labels = train_test_split(
    transcripts, labels, test_size=0.25, random_state=42
)

# Extract embeddings for train and test data
train_embeddings = extract_embeddings(train_transcripts, batch_size=2)  # Smaller batch size to reduce memory usage
test_embeddings = extract_embeddings(test_transcripts, batch_size=2)

# Standardize the embeddings
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Train SVM classifier
print("Training with RBF Kernel:")
svm_classifier_rbf = svm.SVC(kernel='rbf', probability=True)
svm_classifier_rbf.fit(train_embeddings, train_labels)
test_predictions_rbf = svm_classifier_rbf.predict(test_embeddings)

print(f"RBF Kernel Accuracy: {accuracy_score(test_labels, test_predictions_rbf)}")
print("RBF Kernel Classification Report:")
print(classification_report(test_labels, test_predictions_rbf))

# Print the number of trainable parameters (should be zero since all are frozen)
total_params = sum(p.numel() for p in falcon_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")
