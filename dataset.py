import os
import numpy as np
import torch
import cv2
import glob
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
from torchvision import transforms

def get_metadata_or_default(mhd_dir, seriesuid):
    mhd_file = glob.glob(os.path.join(mhd_dir, "**", f"{seriesuid}.mhd"), recursive=True)
    if mhd_file:
        try:
            img = sitk.ReadImage(mhd_file[0])
            origin = np.array(img.GetOrigin())[::-1]
            spacing = np.array(img.GetSpacing())[::-1]
            return origin, spacing
        except Exception as e:
            print(f"Error reading {mhd_file[0]}: {e}")
    print(f"⚠️ Metadata not found for {seriesuid}, using defaults.")
    return np.array([-256.0, -256.0, -500.0]), np.array([1.0, 1.0, 1.0])

class Luna16Dataset(Dataset):
    def __init__(self, mhd_dir, slices_dir, candidates_df, transform, dino_model, device):
        self.mhd_dir = mhd_dir
        self.slices_dir = slices_dir
        self.candidates_df = candidates_df
        self.transform = transform
        self.dino_model = dino_model
        self.device = device
        self.data_info = []
        self.slice_counts = {}

        print("Loading dataset... This may take some time.")
        for subset in range(10):
            subset_slices_path = os.path.join(self.slices_dir, f"subset{subset}")
            if not os.path.exists(subset_slices_path):
                print(f"⚠️ Warning: Subset directory not found: {subset_slices_path}")
                continue

            seriesuids = [d for d in os.listdir(subset_slices_path) if os.path.isdir(os.path.join(subset_slices_path, d))]
            for seriesuid in tqdm(seriesuids, desc=f"Processing subset{subset}"):
                origin, spacing = get_metadata_or_default(self.mhd_dir, seriesuid)

                # Use candidates.csv for SSL pre-training guidance
                series_candidates = self.candidates_df[self.candidates_df["seriesuid"] == seriesuid]
                candidate_z_indices = set()
                for _, row in series_candidates.iterrows():
                    center_z = int(np.rint((row["coordZ"] - origin[2]) / spacing[2]))
                    candidate_z_indices.add(center_z)

                series_slices_path = os.path.join(subset_slices_path, seriesuid)
                slice_files = [f for f in os.listdir(series_slices_path) if f.endswith('.png')]
                self.slice_counts[seriesuid] = len(slice_files)

                for slice_file in slice_files:
                    try:
                        z = int(slice_file.split('_')[1].split('.')[0])
                    except (IndexError, ValueError):
                        print(f"⚠️ Warning: Invalid slice file name: {slice_file}")
                        continue

                    slice_path = os.path.join(series_slices_path, slice_file)
                    if not os.path.exists(slice_path):
                        print(f"⚠️ Warning: Slice file does not exist: {slice_path}")
                        continue

                    # SSL: Use candidates as weak labels
                    label = 1 if z in candidate_z_indices else 0
                    bbox = self.infer_bbox_from_features(slice_path) if label == 1 else [0, 0, 0, 0]

                    self.data_info.append((slice_path, label, bbox))

        print(f"Loaded {len(self.data_info)} slices.")
        if len(self.data_info) == 0:
            print("WARNING: No valid slices found! Attempting fallback...")
            self._load_fallback_data()

    def infer_bbox_from_features(self, slice_path):
        # Load and preprocess the slice
        slice_2d = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        if slice_2d is None:
            print(f"Error loading image {slice_path}, using default bbox")
            return [0, 0, 0, 0]
        slice_2d_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
        slice_tensor = self.transform(slice_2d_rgb).unsqueeze(0).to(self.device)

        # Extract features for the entire slice
        with torch.no_grad():
            features = self.dino_model(slice_tensor)  # Shape: (1, 1536)

        # Split the slice into patches (14x14 for DINOv2 ViT-L/14)
        patch_size = 14
        patches = []
        for i in range(0, 504, patch_size):
            for j in range(0, 504, patch_size):
                patch = slice_tensor[:, :, i:i+patch_size, j:j+patch_size]
                if patch.shape[2] == patch_size and patch.shape[3] == patch_size:
                    patches.append(patch)
        if not patches:
            return [0, 0, 0, 0]
        patches_tensor = torch.cat(patches, dim=0)  # Shape: (num_patches, 3, 14, 14)

        # Extract features for each patch
        with torch.no_grad():
            patch_features = self.dino_model(patches_tensor)  # Shape: (num_patches, 1536)

        # Compute cosine similarity between patch features and the slice feature
        slice_feature = features.squeeze(0)  # Shape: (1536,)
        similarities = torch.nn.functional.cosine_similarity(patch_features, slice_feature.unsqueeze(0), dim=1)

        # Find the patch with the highest similarity
        max_sim_idx = similarities.argmax().item()
        num_patches_per_row = 504 // patch_size  # 36 patches per row
        center_patch_y = (max_sim_idx // num_patches_per_row) * patch_size + patch_size // 2
        center_patch_x = (max_sim_idx % num_patches_per_row) * patch_size + patch_size // 2

        # Infer bounding box around the center (assume 10mm diameter, ~20 pixels)
        box_size = 20  # Fixed size based on typical nodule diameter
        half_box = box_size // 2
        x_min = max(0, center_patch_x - half_box)
        x_max = min(504, center_patch_x + half_box)
        y_min = max(0, center_patch_y - half_box)
        y_max = min(504, center_patch_y + half_box)

        # Normalize bounding box to [0, 1]
        bbox = [x_min / 504.0, y_min / 504.0, (x_max - x_min) / 504.0, (y_max - y_min) / 504.0]
        return bbox

    def _load_fallback_data(self):
        png_files = glob.glob(os.path.join(self.slices_dir, "**", "*.png"), recursive=True)
        print(f"Found {len(png_files)} PNG files in fallback.")
        if len(png_files) > 1000:
            png_files = png_files[:1000]
        for slice_path in png_files:
            self.data_info.append((slice_path, 0, [0, 0, 0, 0]))
        print(f"Added {len(self.data_info)} slices using fallback method.")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        slice_path, label, bbox = self.data_info[idx]
        try:
            slice_2d = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            if slice_2d is None:
                print(f"Error loading image {slice_path}, using zeroed image")
                slice_2d = np.zeros((504, 504), dtype=np.uint8)
            slice_2d_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
            slice_tensor = self.transform(slice_2d_rgb).to(self.device)
            return slice_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(bbox, dtype=torch.float32), slice_path
        except Exception as e:
            print(f"Error processing {slice_path}: {e}")
            return torch.zeros(3, 504, 504), torch.tensor(0, dtype=torch.long), torch.zeros(4, dtype=torch.float32), slice_path