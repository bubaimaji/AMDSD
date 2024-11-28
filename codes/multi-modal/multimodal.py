import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login

# Set your Hugging Face token here
hf_token = "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"
login(token=hf_token)

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

# Configure model loading with 4-bit precision using bitsandbytes
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load Llama 2 model and tokenizer with 4-bit precision
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=quantization_config).to(device)

# Function to extract text features using Llama 2
def extract_text_features(text):
    inputs = llama_tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = llama_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    return last_hidden_state.cpu().numpy()

# Prepare data
texts = combined_data_df['transcript'].tolist()  # Adjust column name as needed
labels = combined_data_df['label'].values  # Adjust column name as needed

# Shuffle and split the data
texts, labels = shuffle(texts, labels, random_state=42)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Extract embeddings for train and test data
train_embeddings = [extract_text_features(text) for text in train_texts]
test_embeddings = [extract_text_features(text) for text in test_texts]

train_embeddings = np.vstack(train_embeddings)
test_embeddings = np.vstack(test_embeddings)

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
