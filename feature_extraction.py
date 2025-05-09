import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from config import mhd_dir, slices_dir, candidates_file, transform, features_dir, device, batch_size

# Load Pre-trained DINOv2 Model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
dinov2_vitl14.eval()

# Load Candidates
df_candidates = pd.read_csv(candidates_file)

# Load Dataset
dataset = Luna16Dataset(mhd_dir, slices_dir, df_candidates, transform, dinov2_vitl14, device)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to extract and save features
def extract_features(loader, split_name):
    features = []
    labels = []
    bboxes = []
    slice_paths = []
    with torch.no_grad():
        for slice_tensors, label_batch, bbox_batch, paths in tqdm(loader, desc=f"Extracting features for {split_name}"):
            slice_tensors = slice_tensors.to(device)
            feature_batch = dinov2_vitl14(slice_tensors).cpu().numpy()
            features.append(feature_batch)
            labels.append(label_batch.numpy())
            bboxes.append(bbox_batch.numpy())
            slice_paths.extend(paths)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    bboxes = np.concatenate(bboxes, axis=0)
    np.save(os.path.join(features_dir, f"{split_name}_features.npy"), features)
    np.save(os.path.join(features_dir, f"{split_name}_labels.npy"), labels)
    np.save(os.path.join(features_dir, f"{split_name}_bboxes.npy"), bboxes)
    np.save(os.path.join(features_dir, f"{split_name}_paths.npy"), np.array(slice_paths))
    print(f"Saved {split_name} features: {features.shape}, labels: {labels.shape}, bboxes: {bboxes.shape}")

# Extract features for all splits
extract_features(train_loader, "train")
extract_features(val_loader, "val")
extract_features(test_loader, "test")