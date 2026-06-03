# Shoe Pair Matching - CNN Image Similarity

A PyTorch implementation of two convolutional neural networks that classify whether two shoes belong to the **same pair** or **different pairs**. Trained and evaluated on a dataset of men's and women's shoes.

---

## The Problem

Given an image containing two shoes side by side, the model must determine whether they form a matching pair. This has real-world applications, for example assisting visually impaired users in their daily independence.

---

## Model Architectures

Two architectures are implemented and compared:

### CNN
Takes a concatenated 448×224 image (two shoes stacked vertically) as input.

- Input: `(N, 3, 448, 224)`
- 4 convolutional blocks: `3 → 32 → 64 → 128 → 256` channels
- Each block: Conv2d (5×5, padding=2) → ReLU → MaxPool (2×2)
- Fully connected: flatten → 100 → 2
- Output: `(N, 2)` - same pair / different pair

### CNNChannel
Splits the input into two 224×224 halves and stacks them along the **channel dimension**, giving the network explicit access to both shoes at every spatial position.

- Input manipulation: `(N, 3, 448, 224)` → `(N, 6, 224, 224)`
- Same convolutional structure as CNN
- Output: `(N, 2)`

![CNNChannel Architecture](assets/cnn_channel_diagram.png)

The key insight: by stacking shoes along the channel axis, convolutional filters can directly compare corresponding pixels from both shoes at the very first layer, detecting color and texture differences immediately, rather than having to learn the comparison implicitly across a large spatial distance.

---

## Results

### Validation Accuracy

| Model | Parameters | Best Val Accuracy |
|-------|-----------|------------------|
| CNN | 11.1M | 77.45% |
| **CNNChannel** | 6.1M | **88.24%** |

CNNChannel outperformed the standard CNN by ~11%, confirming that channel-wise concatenation is a more effective approach for pairwise comparison.

### Test Accuracy (CNNChannel)

| Test Set | Positive Acc | Negative Acc | Average |
|----------|-------------|--------------|---------|
| Women's shoes | 90.00% | 76.67% | **83.33%** |
| Men's shoes | 73.33% | 90.00% | **81.67%** |

Accuracy is tracked separately for positive (same pair) and negative (different pair) samples to prevent model collapse, a scenario where the model achieves decent overall accuracy by always predicting the majority class.

---

## Repository Structure

```
├── shoe_pair_matching_cnn.ipynb   # Main notebook (training, evaluation, visualization)
├── ML_DL_Functions3.py            # CNN and CNNChannel model definitions
└── assets/
    └── cnn_channel_diagram.png    # Architecture diagram
```

> Pre-trained model weights are not included due to file size. To retrain, run the notebook from Section 3.

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision numpy matplotlib Pillow
```

### Running on Google Colab (Recommended)

This project was developed on Google Colab with GPU acceleration. To run:

1. Upload the notebook and files to your Google Drive
2. Open `shoe_pair_matching_cnn.ipynb` in Colab
3. Update `drive_path` to point to your Drive directory
4. Run all cells

### Loading a Trained Model

```python
import torch
from ML_DL_Functions3 import CNNChannel

model = CNNChannel()
model.load_state_dict(torch.load('best_CNNChannel_model.pk', map_location='cpu'))
model.eval()
```

---

## Credits

Implemented by **Asaf Schreiber** and **Shachar Lavi** as part of the *Introduction to Machine Learning* course at **Ben-Gurion University of the Negev**, instructed by **Dr. Nir Shalzinger**.

The assignment framework and dataset were provided by the course staff.
