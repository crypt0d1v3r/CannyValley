import argparse
import asyncio
import os
import io
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import kagglehub
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as vision_datasets
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

app = FastAPI(title="Fake vs Real Image Classifier API")

# Setup directories
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset Metadata
AVAILABLE_DATASETS = [
    "birdy654/cifake-real-and-ai-generated-synthetic-images",
    "Hemg/AI-vs-Real-images",
    "bitmind/AI-vs-Real-Dataset-Images-Proper",
    "Parveshiiii/AI-vs-Real"
]
AVAILABLE_MODELS = ["cnn", "gap"]

# Model Definition
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)

# Data transformations for fixed 224x224 input (CNN)
cnn_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CNN_GAP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Feature Extractor — progressive channel widening with BatchNorm
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 2. Global Average Pooling — captures distributed artifacts
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512), 
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    

# Data transformations for arbitrary-sized inputs (CNN_GAP)
gap_transforms = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='constant'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def is_valid_image(pil_image, min_w, min_h):
    """Return True if the image meets the minimum size requirements."""
    w, h = pil_image.size
    return w >= min_w and h >= min_h


def _save_image(args_tuple):
    """Worker function: checks an image and saves it if it passes the resolution filter."""
    idx, pil_image, class_name, out_class_dir, min_w, min_h, dry_run = args_tuple
    result = {"kept": 0, "skipped": 0, "error": 0}
    try:
        img = pil_image.convert("RGB")
        if not is_valid_image(img, min_w, min_h):
            result["skipped"] += 1
            return result
        if dry_run:
            result["kept"] += 1
            return result
        filename = f"{idx:06d}.jpg"
        save_path = os.path.join(out_class_dir, filename)
        img.save(save_path, format="JPEG", quality=95)
        result["kept"] += 1
    except Exception as e:
        print(f"\n  [WARN] Index {idx} ({class_name}): {e}", file=sys.stderr)
        result["error"] += 1
    return result


def preprocess_dataset(
    dataset_id,
    min_width=256,
    min_height=256,
    output_dir=None,
    split="train",
    num_proc=4,
    dry_run=False,
):
    """Filter low-resolution images from a HuggingFace dataset and export to
    an ImageFolder-compatible directory.

    Args:
        dataset_id: HuggingFace dataset identifier (must be in AVAILABLE_DATASETS).
        min_width:  Minimum image width in pixels.
        min_height: Minimum image height in pixels.
        output_dir: Root output directory. Auto-generated if None.
        split:      Dataset split to process (e.g. "train").
        num_proc:   Number of parallel workers.
        dry_run:    If True, report stats only without saving files.

    Returns:
        Path to the split directory (e.g. "datasets/bitmind_filtered/train").
    """
    if dataset_id not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}. Choose from: {AVAILABLE_DATASETS}")
    if dataset_id == "birdy654/cifake-real-and-ai-generated-synthetic-images":
        raise ValueError(
            "Kaggle dataset does not need preprocessing — it is already a local "
            "ImageFolder. Use one of the HuggingFace datasets instead."
        )

    if output_dir is None:
        safe_name = dataset_id.replace("/", "_")
        output_dir = os.path.join(DATASETS_DIR, f"{safe_name}_filtered")

    print("=" * 60)
    print(f"  Dataset  : {dataset_id}")
    print(f"  Split    : {split}")
    print(f"  Min size : {min_width} x {min_height} px")
    print(f"  Output   : {output_dir}")
    print(f"  Workers  : {num_proc}")
    print(f"  Dry run  : {dry_run}")
    print("=" * 60)

    print(f"\n[1/3] Loading dataset '{dataset_id}' split='{split}' ...")
    ds = load_dataset(dataset_id, split=split, verification_mode="no_checks")

    # Detect label column and class names
    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        label_col = "label"
        label_names = ds.features["label"].names
    elif "binary_label" in ds.features:
        label_col = "binary_label"
        label_names = ["Real", "AI"]  # 0=Real, 1=AI
    else:
        raise ValueError(f"Cannot detect label column. Features: {list(ds.features.keys())}")

    total = len(ds)
    print(f"      Loaded {total:,} rows. Classes: {label_names} (col='{label_col}')")

    class_dirs = {}
    if not dry_run:
        for class_name in label_names:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[class_name] = class_dir
        print(f"\n[2/3] Output directories created under: {output_dir}")
    else:
        print(f"\n[2/3] Dry-run mode — no directories will be created.")

    print(f"\n[3/3] Processing {total:,} images with {num_proc} workers ...\n")

    kept_counts = {name: 0 for name in label_names}
    skipped_counts = {name: 0 for name in label_names}
    error_count = 0

    def task_generator():
        for idx, row in enumerate(ds):
            class_name = label_names[row[label_col]]
            out_class_dir = class_dirs.get(class_name, "")
            yield (idx, row["image"], class_name, out_class_dir, min_width, min_height, dry_run)

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        futures = {
            executor.submit(_save_image, task): task[2]
            for task in task_generator()
        }
        with tqdm(total=total, unit="img", desc="Filtering") as pbar:
            for future in as_completed(futures):
                class_name = futures[future]
                res = future.result()
                kept_counts[class_name] += res["kept"]
                skipped_counts[class_name] += res["skipped"]
                error_count += res["error"]
                pbar.update(1)
                pbar.set_postfix(
                    kept=sum(kept_counts.values()),
                    skipped=sum(skipped_counts.values()),
                )

    total_kept = sum(kept_counts.values())
    total_skipped = sum(skipped_counts.values())

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Class':<10}  {'Kept':>8}  {'Filtered Out':>14}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*14}")
    for name in label_names:
        print(f"  {name:<10}  {kept_counts[name]:>8,}  {skipped_counts[name]:>14,}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*14}")
    print(f"  {'TOTAL':<10}  {total_kept:>8,}  {total_skipped:>14,}")
    if error_count:
        print(f"\n  Errors/corrupt images skipped: {error_count}")
    print(f"\n  Filter threshold : >= {min_width} x {min_height} px")
    print(f"  Reduction        : {total_skipped / total * 100:.1f}% of images removed")
    split_dir = os.path.join(output_dir, split)
    if not dry_run:
        print(f"\n  Output saved to  : {Path(split_dir).resolve()}")
    print("=" * 60)

    return split_dir


class HuggingFaceImageDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace dataset split for use with PyTorch DataLoader."""
    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform
        self.classes = hf_split.features['label'].names

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def kfold_split(data, k=5):
    """Creates k-fold cross-validation splits for the given data.
    
    Args:
        data: PyTorch Dataset to split
        k: Number of folds (default=5)
    
    Yields:
        Tuples of (train_subset, val_subset) for each fold
    """
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    for train_idx, val_idx in kfold.split(indices):
        train_subset = Subset(data, train_idx)
        val_subset = Subset(data, val_idx)
        yield train_subset, val_subset


class TrainRequest(BaseModel):
    dataset_source: str
    model_name: str
    model_type: str = "gap"
    num_epochs: int = 20
    learning_rate: float = 0.001
    holdout_pct: float = 0.1

@app.get("/datasets")
async def get_datasets():
    """Returns a list of available dataset sources for training."""
    return {"available_datasets": AVAILABLE_DATASETS}

@app.get("/models")
async def get_models():
    """Returns a list of locally saved trained models."""
    try:
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainRequest):
    """Downloads dataset, trains model, and saves weights locally."""
    is_local = os.path.isdir(request.dataset_source)
    if not is_local and request.dataset_source not in AVAILABLE_DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset source. Provide a local ImageFolder path or check /datasets for valid options.")
    if request.model_type not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Choose from: {AVAILABLE_MODELS}")

    # Select model class, transforms, and batch size based on model_type
    if request.model_type == "gap":
        ModelClass = CNN_GAP
        active_transforms = gap_transforms
        batch_size = 32
        accumulation_steps = 1
    else:
        ModelClass = CNN
        active_transforms = cnn_transforms
        batch_size = 32
        accumulation_steps = 1

    # 1. Dataset Loading Logic
    print(f"[1/4] Loading dataset: {request.dataset_source} ...", flush=True)
    try:
        if is_local:
            # Local ImageFolder directory — look for a train/ subfolder, fall back to root
            train_dir = os.path.join(request.dataset_source, 'train')
            if not os.path.isdir(train_dir):
                train_dir = request.dataset_source
            train_dataset = vision_datasets.ImageFolder(root=train_dir, transform=active_transforms)
            class_names = train_dataset.classes
        elif request.dataset_source == "birdy654/cifake-real-and-ai-generated-synthetic-images":
            dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
            train_dir = os.path.join(dataset_path, 'train')
            test_dir = os.path.join(dataset_path, 'test')
            if not os.path.exists(train_dir) or not os.path.exists(test_dir):
                raise HTTPException(status_code=500, detail=f"Dataset directories not found at {dataset_path}")
            train_dataset = vision_datasets.ImageFolder(root=train_dir, transform=active_transforms)
            class_names = train_dataset.classes
        elif request.dataset_source == "Hemg/AI-vs-Real-images":
            hf_data = load_dataset("Hemg/AI-vs-Real-images")
            train_dataset = HuggingFaceImageDataset(hf_data['train'], transform=active_transforms)
            class_names = train_dataset.classes
        elif request.dataset_source == "bitmind/AI-vs-Real-Dataset-Images-Proper":
            hf_data = load_dataset("bitmind/AI-vs-Real-Dataset-Images-Proper")
            train_dataset = HuggingFaceImageDataset(hf_data['train'], transform=active_transforms)
            class_names = train_dataset.classes
        elif request.dataset_source == "Parveshiiii/AI-vs-Real":
            hf_data = load_dataset("Parveshiiii/AI-vs-Real", verification_mode="no_checks")
            train_dataset = HuggingFaceImageDataset(hf_data['train'], transform=active_transforms)
            class_names = train_dataset.classes
        else:
            raise HTTPException(status_code=400, detail=f"Dataset Source {request.dataset_source} Unknown")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    print(f"[1/4] Dataset loaded: {len(train_dataset):,} images, classes: {class_names}", flush=True)

    # 2. Model & Training Configuration
    k = 3
    criterion = torch.nn.CrossEntropyLoss()
    weight_decay = 0.01

    # 2b. Holdout split — randomly separate test data before k-fold
    n_total = len(train_dataset)
    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    n_holdout = int(n_total * request.holdout_pct)
    holdout_indices = indices[:n_holdout]
    cv_indices = indices[n_holdout:]
    holdout_set = Subset(train_dataset, holdout_indices)
    cv_dataset = Subset(train_dataset, cv_indices)
    print(f"[2/4] Holdout split: {len(cv_dataset):,} for CV, {len(holdout_set):,} for final test ({request.holdout_pct:.0%})", flush=True)

    # 3. K-Fold Cross-Validation Training Loop
    all_train_errors = np.zeros((k, request.num_epochs))
    all_val_errors = np.zeros((k, request.num_epochs))
    best_val_error = float('inf')
    best_model_state = None
    best_fold = -1
    best_epoch = -1

    print(f"[3/4] Starting {k}-fold CV training for {request.num_epochs} epochs ...", flush=True)
    for fold_num, (train_subset, val_subset) in enumerate(kfold_split(cv_dataset, k=k)):
        print(f"\n--- Fold {fold_num + 1}/{k} ---", flush=True)
        
        # Free previous fold's GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reinitialize model and optimizer for each fold
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate, weight_decay=weight_decay)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        print(f"Training: {len(train_loader)} batches/epoch, batch_size={batch_size}", flush=True)
        for epoch in range(request.num_epochs):
            model.train()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            train_correct = 0
            for i, (image, label) in enumerate(train_loader):
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label) / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                train_correct += (output.argmax(1) == label).sum().item()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step_num = (i + 1) // accumulation_steps
                    if step_num % 10 == 0:
                        print(f"Epoch [{epoch+1}/{request.num_epochs}], Step [{step_num}], Loss: {accumulated_loss:.4f}", flush=True)
                    accumulated_loss = 0.0

            # Flush any remaining accumulated gradients at end of epoch
            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                val_correct = sum(
                    (model(imgs.to(device)).argmax(1) == lbls.to(device)).sum().item()
                    for imgs, lbls in val_loader
                )
            val_error = 1 - val_correct / len(val_subset)
            train_error = 1 - train_correct / len(train_subset)
            all_train_errors[fold_num, epoch] = train_error
            all_val_errors[fold_num, epoch] = val_error
            print(f"  Epoch {epoch+1}/{request.num_epochs} — train_err: {train_error:.4f}, val_err: {val_error:.4f}", flush=True)

            # Track best model across all folds and epochs
            if val_error < best_val_error:
                best_val_error = val_error
                best_model_state = model.state_dict().copy()
                best_fold = fold_num + 1
                best_epoch = epoch + 1
                print(f"  ★ New best model: Fold {best_fold}, Epoch {best_epoch}, Val Error: {best_val_error:.4f}", flush=True)

    # Average across folds
    mean_train_errors = all_train_errors.mean(axis=0)
    mean_val_errors = all_val_errors.mean(axis=0)

    plt.figure()
    plt.plot(range(1, request.num_epochs + 1), mean_train_errors, label='Avg Train Error')
    plt.plot(range(1, request.num_epochs + 1), mean_val_errors, label='Avg Val Error')
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7)
    plt.scatter([best_epoch], [best_val_error], color='r', zorder=5,
                label=f'Best Model (Fold {best_fold}, Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Misclassification Error')
    plt.title(f'{k}-Fold CV Learning Curve')
    plt.legend()
    plot_path = os.path.join(MODELS_DIR, f"{request.model_name}_cv_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved averaged learning curve to {plot_path}", flush=True)
    print(f"Best model from Fold {best_fold}, Epoch {best_epoch} with Val Error: {best_val_error:.4f}", flush=True)

    # Save the best model
    model_path = os.path.join(MODELS_DIR, f"{request.model_name}.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'classes': class_names,
        'model_type': request.model_type
    }, model_path)

    # 4. Holdout Evaluation & Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    print(f"\n[4/4] Evaluating best model on holdout set ...", flush=True)

    model = ModelClass().to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    holdout_loader = DataLoader(holdout_set, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in holdout_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    holdout_accuracy = (all_preds == all_labels).mean()

    print(f"\n{'=' * 60}", flush=True)
    print("  HOLDOUT TEST RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Holdout samples : {len(holdout_set):,}", flush=True)
    print(f"  Holdout accuracy: {holdout_accuracy:.4f}", flush=True)
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names)}", flush=True)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Holdout Confusion Matrix (n={len(holdout_set):,})')
    cm_path = os.path.join(MODELS_DIR, f"{request.model_name}_confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved confusion matrix to {cm_path}", flush=True)
    print(f"{'=' * 60}", flush=True)

    return {
        "message": "Training completed successfully",
        "model_path": model_path,
        "classes": class_names,
        "holdout_accuracy": float(holdout_accuracy),
        "confusion_matrix": cm.tolist()
    }


@app.post("/predict")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    """Accepts an image file and a model name, returns class prediction."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
         raise HTTPException(status_code=404, detail="Model not found. Use /models to see available models.")
    
    # Load Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
    try:
        # Prepare model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_type = checkpoint.get('model_type', 'gap')
        ModelClass = CNN_GAP if model_type == 'gap' else CNN
        active_transforms = gap_transforms if model_type == 'gap' else cnn_transforms
        model = ModelClass().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        class_names = checkpoint.get('classes', ["FAKE", "REAL"]) # fallback if older format

        # Transform and predict
        input_tensor = active_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = class_names[predicted_idx.item()]
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence.item()),
            "probabilities": {class_names[i]: float(probabilities[0][i].item()) for i in range(len(class_names))}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake vs Real Image Classifier")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Training Command Setup
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--dataset", type=str, required=True, help="Dataset source name")
    train_parser.add_argument("--name", type=str, required=True, help="Name to save the model as")
    train_parser.add_argument("--model-type", type=str, default="gap", choices=AVAILABLE_MODELS, help="Model architecture to use")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--holdout-pct", type=float, default=0.1, help="Fraction of data to hold out for final testing (default: 0.1)")

    # Prediction Command Setup
    predict_parser = subparsers.add_parser("predict", help="Predict an image")
    predict_parser.add_argument("--name", type=str, required=True, help="Name of the trained model to use")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to the image file")

    # Preprocess Command Setup
    preprocess_parser = subparsers.add_parser("preprocess", help="Filter low-res images from a HuggingFace dataset")
    preprocess_parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset identifier")
    preprocess_parser.add_argument("--min-width", type=int, default=256, help="Minimum image width (default: 256)")
    preprocess_parser.add_argument("--min-height", type=int, default=256, help="Minimum image height (default: 256)")
    preprocess_parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-generated if omitted)")
    preprocess_parser.add_argument("--split", type=str, default="train", help="Dataset split to process (default: train)")
    preprocess_parser.add_argument("--num-proc", type=int, default=4, help="Number of parallel workers (default: 4)")
    preprocess_parser.add_argument("--dry-run", action="store_true", help="Report stats only, do not save images")

    args = parser.parse_args()

    if args.command == "train":
        asyncio.run(train_model(TrainRequest(
            dataset_source=args.dataset,
            model_name=args.name,
            model_type=args.model_type,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            holdout_pct=args.holdout_pct
        )))
    elif args.command == "predict":
        with open(args.image, "rb") as f:
            upload = UploadFile(filename=os.path.basename(args.image), file=io.BytesIO(f.read()))
        result = asyncio.run(predict(model_name=args.name, file=upload))
        print(result)
    elif args.command == "preprocess":
        preprocess_dataset(
            dataset_id=args.dataset,
            min_width=args.min_width,
            min_height=args.min_height,
            output_dir=args.output_dir,
            split=args.split,
            num_proc=args.num_proc,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()