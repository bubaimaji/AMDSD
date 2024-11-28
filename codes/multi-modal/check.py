import numpy as np
import pandas as pd
import os


# Load the .npy file
data = np.load('/home/suhita/Documents/multimodal/multi-modal/features/train/text_features.npy')

# Check the shape of the data
print(f"Shape of the data: {data.shape}")

#print(data[:5])