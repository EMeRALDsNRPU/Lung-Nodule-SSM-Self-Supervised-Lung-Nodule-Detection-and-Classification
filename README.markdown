# Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification

This repository contains the source code for the paper *"Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification in Thoracic CT Images"* by Muniba Noreen and Furqan Shaukat, published in 2025. The project implements a fully self-supervised learning approach using DINOv2 for lung nodule detection and classification on the LUNA16 dataset, achieving an accuracy of 98.37% with a Random Forest classifier. Bounding boxes are inferred using feature similarity from DINOv2 representations.

## Overview

The methodology consists of two stages:
1. **Feature Extraction**: Pre-trained DINOv2 model extracts robust features from lung CT slices, guided by `candidates.csv`.
2. **Nodule Detection and Classification**:
   - Uses DINOv2 feature similarity to infer nodule centers and bounding boxes in a self-supervised manner.
   - Fine-tunes a `DinoDetector` model for classification and bounding box regression using pseudo-labels.
   - Evaluates with classical classifiers (Random Forest, Decision Tree, KNN) on extracted features.

## Requirements

- Python 3.8+
- NVIDIA GPU (tested on RTX 3090)
- Dependencies listed in `requirements.txt`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the LUNA16 dataset, which consists of 888 CT scans. You need to:
1. Download the dataset from the [LUNA16 challenge website](https://luna16.grand-challenge.org/).
2. Preprocess the CT scans into 2D PNG slices.
3. Place the `.mhd` files in `C:/Users/naeem/Desktop/SSL_REPO/DINO/LUNA16` and PNG slices in `C:/Users/naeem/Desktop/SSL_REPO/DINO/LUNA_png`.
4. Ensure the `candidates.csv` file is at `C:/Users/naeem/Desktop/SSL_REPO/DINO/candidates.csv`.

## Project Structure

```
LungNodule-SSM/
│
├── config.py               # Configuration and hyperparameters
├── dataset.py              # Dataset loading and preprocessing
├── model.py                # DinoDetector model definition
├── feature_extraction.py   # Stage 1: Extract DINOv2 features
├── train.py                # Stage 2: Train DinoDetector and evaluate with classical classifiers
├── requirements.txt        # Dependencies
├── paper_2025.pdf            # The published paper
└── README.md               # This file
```

## Usage

1. **Extract Features**:
   Run the feature extraction script to generate DINOv2 features for the dataset.
   ```bash
   python feature_extraction.py
   ```
   This will save features in the `features_dir` specified in `config.py`.

2. **Train and Evaluate Model**:
   Train the `DinoDetector` model and evaluate it, including classical classifiers.
   ```bash
   python train.py
   ```
   This will output performance metrics for the `DinoDetector` and classical classifiers, save annotated images in `annotated_dir`, and save model checkpoints in `checkpoint_dir`.



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{noreen2025lung,
  title={Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification in Thoracic CT Images},
  author={Noreen, Muniba and Shaukat, Furqan},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
