
---

# AFPDeepPred: Antifreeze Protein Prediction

This repository contains the **PyTorch implementation of AFPDeepPred**, a novel multi-modal deep learning framework designed to predict **Antifreeze Proteins (AFPs)**.
The model integrates:

* **Evolutionary Scale Model (ESM-2)** to capture *global sequence information*, and
* **Chaos Game Representation (CGR)** to highlight *local motif distributions*.

Feature fusion is accomplished using a **Bilinear Attention Network (BAN)**, enabling high-accuracy AFP prediction.

---

## 🚀 Features

* **Multi-Modal Approach**: Combines ESM-2 embeddings (global features) and CGR images (local features) for a comprehensive protein representation.
* **Attention-Based Fusion**: Utilizes a Bilinear Attention Network (BAN) to effectively fuse heterogeneous features and capture fine-grained cross-modal interactions.
* **Handles Class Imbalance**: Employs **Focal Loss** to address the inherent imbalance in AFP datasets, improving sensitivity on minority classes.
* **Modular & Reproducible**: Codebase is organized into clear, reusable scripts for easy configuration, training, and evaluation.

---

## 📂 Project Structure

```
AFPDeepPred/
├── data/
│   ├── train.csv         # Training dataset (download from Zenodo)
│   └── val.csv           # Reviewed independent dataset (split from train.csv)
├── models/
│   └── best_model.pth    # Pretrained model weights (download from Zenodo)
├── config.py             # Configuration file (paths, hyperparameters)
├── data_loader.py        # Data loading and preprocessing logic
├── model.py              # Model architecture (AFPDeepPred, BANLayer, FocalLoss)
├── train.py              # Script to train the model
├── evaluate.py           # Script to evaluate the trained model
└── requirements.txt      # List of required Python packages
```

---

## 📥 Data and Pretrained Model

You can download the training dataset (`train.csv`) and pretrained model (`best_model.pth`) directly from **Zenodo**:

🔗 [Download from Zenodo](https://zenodo.org/uploads/16886732)

After downloading:

* Place `train.csv` into the `data/` directory
* Place `best_model.pth` into the `models/` directory



---

## 🛠️ Setup and Installation

1. **Clone the repository**

```bash
git clone https://github.com/XingqiaoLin/AFP.git
cd AFPDeepPred
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. Configuration

Edit `config.py` to set file paths and training hyperparameters:

```python
# --- File Paths ---
TRAIN_CSV_PATH = 'data/train.csv'
VAL_CSV_PATH   = 'data/val.csv'
MODEL_SAVE_DIR = 'models'
BEST_MODEL_PATH = f'{MODEL_SAVE_DIR}/best_model.pth'

# --- Training Hyperparameters ---
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

### 2. Model Training

Train the model from scratch:

```bash
python train.py
```

The training script will automatically save the model with the best validation **AUC** to `models/best_model.pth`.

### 3. Model Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

The script reports the following metrics:

* Accuracy
* Sensitivity
* Specificity
* Matthews Correlation Coefficient (MCC)
* Area Under the Curve (AUC)
