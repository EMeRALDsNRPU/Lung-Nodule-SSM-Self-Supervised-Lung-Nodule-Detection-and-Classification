import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import cv2
from tqdm import tqdm
import joblib
from model import DinoDetector
import pandas as pd
from config import mhd_dir, slices_dir, candidates_file, transform, device, batch_size, learning_rate, weight_decay, num_epochs, checkpoint_dir, annotated_dir, features_dir

# Load Pre-trained DINOv2 Model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
dinov2_vitl14.eval()

# Compute feature_dim
dummy_slice = np.zeros((504, 504, 3), dtype=np.uint8)
dummy_tensor = transform(dummy_slice).unsqueeze(0).to(device)
with torch.no_grad():
    dummy_feature = dinov2_vitl14(dummy_tensor).flatten()
feature_dim = dummy_feature.shape[0]

# Load Candidates
df_candidates = pd.read_csv(candidates_file)

# Load Dataset
dataset = Luna16Dataset(mhd_dir, slices_dir, df_candidates, transform, dinov2_vitl14, device)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
model = DinoDetector(feature_dim).to(device)

# Loss Functions and Optimizer
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Function to Draw Bounding Boxes
def draw_bboxes_on_slice(slice_path, bbox_pred, bbox_gt, output_path):
    slice_2d = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if slice_2d is None:
        print(f"Error loading image {slice_path}")
        return
    slice_2d_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)
    x_min_pred = int(bbox_pred[0] * 504)
    y_min_pred = int(bbox_pred[1] * 504)
    w_pred = int(bbox_pred[2] * 504)
    h_pred = int(bbox_pred[3] * 504)
    x_max_pred = x_min_pred + w_pred
    y_max_pred = y_min_pred + h_pred
    x_min_gt = int(bbox_gt[0] * 504)
    y_min_gt = int(bbox_gt[1] * 504)
    w_gt = int(bbox_gt[2] * 504)
    h_gt = int(bbox_gt[3] * 504)
    x_max_gt = x_min_gt + w_gt
    y_max_gt = y_min_gt + h_gt
    cv2.rectangle(slice_2d_rgb, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (0, 255, 0), 2)
    cv2.rectangle(slice_2d_rgb, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 0, 255), 2)
    cv2.imwrite(output_path, slice_2d_rgb)

# Training Loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    for slice_tensors, labels, bboxes, slice_paths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        slice_tensors = slice_tensors.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            features = dinov2_vitl14(slice_tensors).flatten(1)
        class_output, bbox_output = model(features)
        
        loss_class = criterion_class(class_output, labels)
        mask = labels == 1
        loss_bbox = criterion_bbox(bbox_output[mask], bboxes[mask]) if mask.sum() > 0 else torch.tensor(0.0).to(device)
        loss = loss_class + loss_bbox
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
        _, predicted = torch.max(class_output, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for slice_tensors, labels, bboxes, slice_paths in val_loader:
            slice_tensors = slice_tensors.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            features = dinov2_vitl14(slice_tensors).flatten(1)
            class_output, bbox_output = model(features)
            loss_class = criterion_class(class_output, labels)
            mask = labels == 1
            loss_bbox = criterion_bbox(bbox_output[mask], bboxes[mask]) if mask.sum() > 0 else torch.tensor(0.0).to(device)
            loss = loss_class + loss_bbox
            total_val_loss += loss.item()
            
            _, predicted = torch.max(class_output, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            
            for i in range(len(predicted)):
                if predicted[i].item() == 1 or labels[i].item() == 1:
                    slice_path = slice_paths[i]
                    bbox_pred = bbox_output[i].cpu().numpy()
                    bbox_gt = bboxes[i].cpu().numpy()
                    output_filename = f"annotated_epoch{epoch+1}_{os.path.basename(slice_path)}"
                    output_path = os.path.join(annotated_dir, output_filename)
                    draw_bboxes_on_slice(slice_path, bbox_pred, bbox_gt, output_path)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    checkpoint_path = os.path.join(checkpoint_dir, f"dino_detector_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

# Save Final Model
final_model_path = os.path.join(checkpoint_dir, "dino_detector_final.pth")
torch.save(model.state_dict(), final_model_path)
print("Final model saved successfully!")

# Evaluate with Classical Classifiers (using pseudo-labels)
train_features = np.load(os.path.join(features_dir, "train_features.npy"))
train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))
val_features = np.load(os.path.join(features_dir, "val_features.npy"))
val_labels = np.load(os.path.join(features_dir, "val_labels.npy"))
test_features = np.load(os.path.join(features_dir, "test_features.npy"))
test_labels = np.load(os.path.join(features_dir, "test_labels.npy"))

classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, clf in classifiers.items():
    print(f"Evaluating {name}...")
    clf.fit(train_features, train_labels)
    val_preds = clf.predict(val_features)
    if hasattr(clf, "predict_proba"):
        val_probs = clf.predict_proba(val_features)[:, 1]
    else:
        val_probs = val_preds
    test_preds = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        test_probs = clf.predict_proba(test_features)[:, 1]
    else:
        test_probs = test_preds
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, zero_division=0)
    val_recall = recall_score(val_labels, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    val_auc = roc_auc_score(val_labels, val_probs)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_auc = roc_auc_score(test_labels, test_probs)
    results[name] = {"accuracy": test_accuracy, "precision": test_precision, "recall": test_recall, "f1": test_f1, "auc": test_auc}
    print(f"{name} Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

print("\nResults on LUNA16 Dataset (Test Set):")
print("| Classifier     | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC (%) |")
print("|----------------|--------------|---------------|------------|--------------|---------|")
for name, metrics in results.items():
    print(f"| {name:<14} | {metrics['accuracy']*100:>12.2f} | {metrics['precision']*100:>13.2f} | "
          f"{metrics['recall']*100:>10.2f} | {metrics['f1']*100:>12.2f} | {metrics['auc']*100:>7.2f} |")