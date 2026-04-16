python main.py train --dataset bitmind/AI-vs-Real-Dataset-Images-Proper --name my_model --epochs 50
python main.py predict --name my_model --image path/to/image.jpg

**Train with GMP model (default):**
```bash
python main.py train --dataset "birdy654/cifake-real-and-ai-generated-synthetic-images" --name my_gmp_model
```

**Train with standard CNN:**
```bash
python main.py train --dataset "birdy654/cifake-real-and-ai-generated-synthetic-images" --name my_cnn_model --model-type cnn
```

**Train with a HuggingFace dataset, custom epochs/lr:**
```bash
python main.py train --dataset "bitmind/AI-vs-Real-Dataset-Images-Proper" --name bitmind_gmp --model-type gmp --epochs 20 --lr 0.0005
```

**Preprocess a HuggingFace dataset (filter out images smaller than 256x256):**
```bash
python main.py preprocess --dataset "bitmind/AI-vs-Real-Dataset-Images-Proper" --min-width 256 --min-height 256
```

**Dry run (see filtering stats without saving files):**
```bash
python main.py preprocess --dataset "bitmind/AI-vs-Real-Dataset-Images-Proper" --dry-run
```

**Train on the preprocessed/filtered dataset using the local ImageFolder output:**
```bash
python main.py train --dataset "birdy654/cifake-real-and-ai-generated-synthetic-images" --name bitmind_filtered_gmp --epochs 20
```
Note: After preprocessing, the filtered images are saved to datasets/bitmind_AI-vs-Real-Dataset-Images-Proper_filtered/train/
which is structured as an ImageFolder. You can point the cifake Kaggle dataset loader at it, or load it manually.

**Predict (model type is loaded automatically from the checkpoint):**
```bash
python main.py predict --name my_gmp_model --image /path/to/image.jpg
```