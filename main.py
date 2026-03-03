import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import kagglehub
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as vision_datasets
import torch
import numpy as np
from PIL import Image

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

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    class_names = train_dataset.classes
    
    # 2. Model Initialization
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    weight_decay = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate, weight_decay=weight_decay)

    # 3. Training Loop
    model.train()
    print(f"Starting training for {request.num_epochs} epochs with {len(train_loader)} batches per epoch...")
    for epoch in range(request.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{request.num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    # Save the model
    model_path = os.path.join(MODELS_DIR, f"{request.model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
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
        model = CNN().to(device)
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
