import numpy as np
import pandas as pd

# Load the combined datasets from NPY files
train_data = np.load("/home/suhita/Documents/multimodal/multi-modal/segmented_audio/train_segmented_df.npy", allow_pickle=True).item()
train_combined_df = pd.DataFrame(train_data)

# Check the class distribution
class_counts = train_combined_df['label'].value_counts()
print("Class distribution before balancing:")
print(class_counts)

# Identify the minority and majority classes
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Oversample the minority class
minority_df = train_combined_df[train_combined_df['label'] == minority_class]
majority_df = train_combined_df[train_combined_df['label'] == majority_class]

# Calculate the number of samples needed to balance the dataset
num_samples_to_add = majority_df.shape[0] - minority_df.shape[0]

# Randomly sample with replacement from the minority class
oversampled_minority_df = minority_df.sample(n=num_samples_to_add, replace=True, random_state=42)

# Concatenate the oversampled minority class with the majority class
balanced_train_df = pd.concat([majority_df, minority_df, oversampled_minority_df], ignore_index=True)

# Shuffle the balanced dataset
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the class distribution after balancing
balanced_class_counts = balanced_train_df['label'].value_counts()
print("Class distribution after balancing:")
print(balanced_class_counts)

# Save the balanced dataset to a new NPY file
balanced_train_data = balanced_train_df.to_dict('list')
np.save("/home/suhita/Documents/multimodal/multi-modal/segmented_audio/balanced_train_combined_df.npy", balanced_train_data)

# Optionally, save the balanced dataset to a CSV file
balanced_train_df.to_csv("/home/suhita/Documents/multimodal/multi-modal/segmented_audio/balanced_train_combined_df.csv", index=False)

print("Balanced dataset saved successfully.")
