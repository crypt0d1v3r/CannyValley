import argparse
import asyncio
import os
import io
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
    "bitmind/AI-vs-Real-Dataset-Images-Proper"
]

# Model Definition
# class CNN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2),
#             torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2),
#             torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2),
#             torch.nn.Flatten(),
#             torch.nn.Linear(64 * 28 * 28, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 2)
#         )

#     def forward(self, x):
#         return self.model(x)

# # Data transformations
# data_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

class CNN_GMP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Feature Extractor (The "Sliding Window")
        # This part doesn't care about input dimensions.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 2. Global Max Pooling
        # This forces ANY spatial dimension down to a 1x1 grid.
        # If the input to this layer is (Batch, 64, 100, 100), it becomes (Batch, 64, 1, 1).
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Notice the input is now exactly 64 (the number of channels), 
            # NOT 64 * 28 * 28.
            nn.Linear(64, 512), 
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_max_pool(x)
        x = self.classifier(x)
        return x
    

# Data transformations for arbitrary-sized inputs
data_transforms = transforms.Compose([
    # REMOVED: transforms.Resize(256)
    # REMOVED: transforms.CenterCrop(224)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    num_epochs: int = 50
    learning_rate: float = 0.001

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
    if request.dataset_source not in AVAILABLE_DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset source. Check /datasets for valid options.")

    # 1. Dataset Loading Logic
    dataset_path = DATASETS_DIR
    try:
        if request.dataset_source == "birdy654/cifake-real-and-ai-generated-synthetic-images":
            dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
        elif request.dataset_source == "Hemg/AI-vs-Real-images":
            dataset_path = load_dataset("Hemg/AI-vs-Real-images")
            raise HTTPException(status_code=501, detail="HuggingFace dataset custom export mapping not supported yet.")
        elif request.dataset_source == "bitmind/AI-vs-Real-Dataset-Images-Proper":
            dataset_path = load_dataset("bitmind/AI-vs-Real-Dataset-Images-Proper")
            raise HTTPException(status_code=501, detail="HuggingFace dataset custom export mapping not supported yet.")
        else:
             raise HTTPException(status_code=400, detail=f"Dataset Source {request.dataset_source} Unknown")
    except HTTPException:
        raise
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    train_dir = os.path.join(dataset_path, 'train')
    
    test_dir = os.path.join(dataset_path , 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise HTTPException(status_code=500, detail=f"Dataset directories not found at {dataset_path}")

    train_dataset = vision_datasets.ImageFolder(root=train_dir, transform=data_transforms)
    class_names = train_dataset.classes
    
    # 2. Model & Training Configuration
    k = 5
    criterion = torch.nn.CrossEntropyLoss()
    weight_decay = 0.01
    accumulation_steps = 32

    # 3. K-Fold Cross-Validation Training Loop
    all_train_errors = np.zeros((k, request.num_epochs))
    all_val_errors = np.zeros((k, request.num_epochs))
    best_val_error = float('inf')
    best_model_state = None
    best_fold = -1
    best_epoch = -1

    for fold_num, (train_subset, val_subset) in enumerate(kfold_split(train_dataset, k=k)):
        print(f"\n--- Fold {fold_num + 1}/{k} ---")

        # Reinitialize model and optimizer for each fold
        model = CNN_GMP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate, weight_decay=weight_decay)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        print(f"Starting training for {request.num_epochs} epochs ({len(train_loader)} images, accumulating {accumulation_steps} steps)...")
        for epoch in range(request.num_epochs):
            model.train()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            for i, (image, label) in enumerate(train_loader):
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label) / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step_num = (i + 1) // accumulation_steps
                    if step_num % 10 == 0:
                        print(f"Epoch [{epoch+1}/{request.num_epochs}], Step [{step_num}], Loss: {accumulated_loss:.4f}")
                    accumulated_loss = 0.0

            # Flush any remaining accumulated gradients at end of epoch
            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                train_correct = sum(
                    (model(imgs.to(device)).argmax(1) == lbls.to(device)).sum().item()
                    for imgs, lbls in train_loader
                )
                val_correct = sum(
                    (model(imgs.to(device)).argmax(1) == lbls.to(device)).sum().item()
                    for imgs, lbls in val_loader
                )
            val_error = 1 - val_correct / len(val_subset)
            all_train_errors[fold_num, epoch] = 1 - train_correct / len(train_subset)
            all_val_errors[fold_num, epoch] = val_error

            # Track best model across all folds and epochs
            if val_error < best_val_error:
                best_val_error = val_error
                best_model_state = model.state_dict().copy()
                best_fold = fold_num + 1
                best_epoch = epoch + 1
                print(f"New best model: Fold {best_fold}, Epoch {best_epoch}, Val Error: {best_val_error:.4f}")

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
    print(f"Saved averaged learning curve to {plot_path}")
    print(f"Best model from Fold {best_fold}, Epoch {best_epoch} with Val Error: {best_val_error:.4f}")

    # Save the best model
    model_path = os.path.join(MODELS_DIR, f"{request.model_name}.pth")
    torch.save({
        'model_state_dict': best_model_state,
        'classes': class_names
    }, model_path)

    return {"message": "Training completed successfully", "model_path": model_path, "classes": class_names}


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
        model = CNN_GMP().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        class_names = checkpoint.get('classes', ["FAKE", "REAL"]) # fallback if older format
        
        # Transform and predict
        input_tensor = data_transforms(image).unsqueeze(0).to(device)
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
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Prediction Command Setup
    predict_parser = subparsers.add_parser("predict", help="Predict an image")
    predict_parser.add_argument("--name", type=str, required=True, help="Name of the trained model to use")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "train":
        asyncio.run(train_model(TrainRequest(
            dataset_source=args.dataset,
            model_name=args.name,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )))
    elif args.command == "predict":
        with open(args.image, "rb") as f:
            upload = UploadFile(filename=os.path.basename(args.image), file=io.BytesIO(f.read()))
        result = asyncio.run(predict(model_name=args.name, file=upload))
        print(result)
    else:
        parser.print_help()