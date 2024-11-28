import torch
import torch.nn as nn
from transformers import GPT2Model, WhisperModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the custom multimodal model class
class MultimodalModel(nn.Module):
    def __init__(self, whisper_model_name="openai/whisper-medium", gpt2_model_name="gpt2"):
        super(MultimodalModel, self).__init__()
        
        # Load pre-trained Whisper model (using only the encoder)
        self.audio_model = WhisperModel.from_pretrained(whisper_model_name).encoder
        self.audio_model.requires_grad_(True)  # Ensure all parameters are trainable
        
        # Load pre-trained GPT-2 model
        self.gpt2_model = GPT2Model.from_pretrained(gpt2_model_name)
        self.gpt2_model.requires_grad_(True)  # Ensure all parameters are trainable

        # Fully connected layer to map speech embedding to GPT-2 hidden space
        self.fc_layer = nn.Linear(self.audio_model.config.hidden_size, self.gpt2_model.config.n_embd)

    def forward(self, audio_input, transcript_input):
        # Process audio input with Whisper encoder
        audio_features = self.audio_model(audio_input).last_hidden_state
        speech_context = torch.relu(self.fc_layer(audio_features.mean(dim=1)))  # Mean pooling and FC layer
        speech_context = speech_context.unsqueeze(1)  # Add sequence dimension

        # Process text input with GPT-2 model
        text_embeddings = self.gpt2_model.wte(transcript_input)

        # Concatenate speech context with text embeddings
        combined_embeddings = torch.cat([speech_context, text_embeddings], dim=1)

        # Adjust attention mask to consider speech context
        extended_attention_mask = torch.cat([torch.ones(speech_context.size(0), 1).to(speech_context.device), transcript_input.attention_mask], dim=1)

        # Pass through GPT-2 transformer layers
        outputs = self.gpt2_model(inputs_embeds=combined_embeddings, attention_mask=extended_attention_mask)
        
        return outputs.last_hidden_state

# Initialize the model
model = MultimodalModel().to(device)

# Calculate total trainable parameters
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters in the entire model: {total_trainable_params}")
