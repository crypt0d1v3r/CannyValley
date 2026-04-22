
# Fake vs Real Image Classifier — Command Line Interface

This project provides a command-line interface (CLI) for training, evaluating, and using deep learning models to classify images as AI-generated (fake) or real. The CLI is implemented in `main.py` and supports dataset preprocessing, model training, and image prediction.

## Setup

1. **Install dependencies:**
	 ```bash
	 pip install -r requirements.txt
	 ```
2. **(Optional) Activate your virtual environment:**
	 ```bash
	 source venv/bin/activate
	 ```

## CLI Usage

Run the CLI with:
```bash
python main.py <command> [options]
```

### Commands

#### 1. Preprocess a HuggingFace Dataset
Filter out low-resolution images and export to a local ImageFolder structure.

**Example:**
```bash
python main.py preprocess \
	--dataset bitmind/AI-vs-Real-Dataset-Images-Proper \
	--min-width 256 \
	--min-height 256 \
	--output-dir datasets/bitmind_AI-vs-Real_filtered \
	--split train \
	--num-proc 8
```
- `--dataset` (required): HuggingFace dataset ID
- `--min-width`, `--min-height`: Minimum image size (default: 256)
- `--output-dir`: Output directory (auto-generated if omitted)
- `--split`: Dataset split (default: train)
- `--num-proc`: Number of parallel workers (default: 4)
- `--dry-run`: Only report stats, do not save images

#### 2. Train a Model
Train a model on one or more datasets (local or HuggingFace) and save the best model.

**Example (single dataset):**
```bash
python main.py train \
	--dataset datasets/bitmind_AI-vs-Real_filtered \
	--name mymodel \
	--model-type gap \
	--epochs 20 \
	--lr 0.0005 \
	--holdout-pct 0.1
```

**Example (multiple datasets):**
```bash
python main.py train \
	--dataset datasets/bitmind_AI-vs-Real_filtered datasets/Parveshiiii_AI-vs-Real_filtered \
	--name combined_model \
	--model-type gap \
	--epochs 30
```
- `--dataset` (required): One or more dataset paths or HuggingFace IDs
- `--name` (required): Name for the saved model
- `--model-type`: Model architecture (`gap` or `cnn`, default: gap)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--holdout-pct`: Fraction of data for holdout test (default: 0.1)

#### 3. Predict an Image
Classify a single image using a trained model.

**Example:**
```bash
python main.py predict \
	--name mymodel \
	--image path/to/image.jpg
```
- `--name` (required): Name of the trained model to use
- `--image` (required): Path to the image file

**Sample Output:**
```
{'prediction': 'Real', 'confidence': 0.98, 'probabilities': {'Real': 0.98, 'AI': 0.02}}
```

---

## Additional Notes
- Trained models are saved in the `models/` directory as `.pth` files.
- Preprocessed datasets are saved in the `datasets/` directory.
- Learning curves and confusion matrices are saved as PNG images in `models/`.
- For available datasets and models, use the FastAPI endpoints (`/datasets`, `/models`) if running as an API.

## Troubleshooting
- Ensure all dependencies are installed and the correct Python environment is active.
- For HuggingFace datasets, you may need to be logged in with `huggingface-cli login`.
- For Kaggle datasets, ensure you have Kaggle API credentials set up.

## License
See `LICENSE` file if present.