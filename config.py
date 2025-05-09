import os
import torch
from torchvision import transforms

# Define Paths
mhd_dir = r"path to your dataset/LUNA16"  # Directory with original .mhd files for metadata
slices_dir = r"path to your dataset/LUNA_png"  # Directory with 2D .png slices
candidates_file = r"path to your/candidates.csv"  # Candidate locations for SSL
checkpoint_dir = r"path to your/checkpoints"  # Directory to save model checkpoints
annotated_dir = r"path to your/annotated_bbox"  # Directory to save annotated slices
features_dir = r"path to your/DINO/features"  # Directory to save extracted features
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters as per the paper
learning_rate = 0.0001
batch_size = 32
weight_decay = 0.01
num_epochs = 100

# Transformations for DINOv2 (expects 3-channel RGB images, resized to 504x504 as per paper)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((504, 504)),  # Resize to 504x504 as per paper
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Patch size for DINOv2 (ViT-L/14 uses 14x14 patches)
patch_size = 14
num_patches = (504 // patch_size) ** 2  # 1296 patches (36x36 grid)