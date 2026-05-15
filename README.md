UE24CS645BC2_PES1PG25CS047_Fashion_MNIST_CNN

# CNN from Scratch - Fashion MNIST Classification

## Project Overview

This project implements a **Convolutional Neural Network (CNN) from scratch using only NumPy** for Fashion-MNIST image classification. The entire CNN architecture is built from basic principles without relying on deep learning frameworks.

The CNN consists of:
- **Convolution Layer** - Extracts local features from images
- **ReLU Activation** - Introduces non-linearity
- **Max Pooling Layer** - Reduces spatial dimensions
- **Fully Connected Layer** - Softmax classifier for 10 classes

---

## Dataset

### Fashion-MNIST

**70,000 grayscale images** of clothing and accessories:

**10 Classes:**
- 0 → T-shirt/top
- 1 → Trouser
- 2 → Pullover
- 3 → Dress
- 4 → Coat
- 5 → Sandal
- 6 → Shirt
- 7 → Sneaker
- 8 → Bag
- 9 → Ankle boot

**Image Specifications:**
- Image Size: **28 × 28 pixels** (grayscale)
- Training Samples: 60,000 images
- Testing Samples: 10,000 images
- Normalized Range: [0, 1]

---

## CNN Architecture

```
Input (28 × 28 × 1)
     ↓
Convolution Layer (16 filters, 3×3)
     Output: 26 × 26 × 16
     ↓
ReLU Activation
     ↓
Max Pooling (2 × 2)
     Output: 13 × 13 × 16
     ↓
Flatten
     Output: 2,704 neurons
     ↓
Fully Connected + Softmax
     Output: 10 classes
```

---

## Features Implemented

✓ Convolution Operation (Manual 2D convolution)
✓ ReLU Activation Function
✓ Max Pooling Layer
✓ Softmax Classifier
✓ Forward Propagation
✓ Backpropagation with Gradient Descent
✓ Cross-Entropy Loss
✓ Accuracy Calculation
✓ Epoch vs Accuracy Graph
✓ Epoch vs Loss Graph

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Learning Rate | 0.01 |
| Batch Size | 1 (SGD) |
| Filters | 16 |
| Filter Size | 3 × 3 |
| Training Samples | 5,000 |
| Test Samples | 500 |

---

## Performance Results

### Training Progress

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.7529 | 74.28% |
| 2 | 0.4651 | 84.50% |
| 3 | 0.3942 | 86.86% |
| 4 | 0.3502 | 88.24% |
| 5 | 0.3169 | 89.24% |

### Final Results

```
Test Accuracy: 85.00%
Training Accuracy (Epoch 5): 89.24%
Final Loss: 0.3169
```

---

## Accuracy Analysis

### Training Accuracy Progression

| Epoch | Accuracy | Improvement |
|-------|----------|-------------|
| 1 | 74.28% | - |
| 2 | 84.50% | +10.22% |
| 3 | 86.86% | +2.36% |
| 4 | 88.24% | +1.38% |
| 5 | 89.24% | +1.00% |

**Observations:**
- Training accuracy increases consistently across all epochs
- 15.0% improvement from epoch 1 to 5
- Gap between training (89.24%) and test (85.00%) accuracy is 4.24% - good generalization
- Model successfully learns meaningful features for clothing classification

---

## Loss Analysis

### Training Loss Progression

| Epoch | Loss | Reduction |
|-------|------|-----------|
| 1 | 0.7529 | - |
| 2 | 0.4651 | -38.2% |
| 3 | 0.3942 | -15.2% |
| 4 | 0.3502 | -11.1% |
| 5 | 0.3169 | -9.5% |

**Observations:**
- Loss decreases monotonically across all epochs
- Largest reduction between epochs 1-2 (38.2% decrease)
- 57.9% total reduction in loss from epoch 1 to 5
- Decreasing loss indicates improved model predictions

---

## Graphs Generated

### 1. Epoch vs Accuracy
- Shows training accuracy improvement during CNN training
- Observed Trend: Steady increase from 74.28% to 89.24%
- File: `epoch_vs_accuracy.png`

### 2. Epoch vs Loss
- Shows reduction in cross-entropy loss during training
- Observed Trend: Steady decrease from 0.7529 to 0.3169
- File: `epoch_vs_loss.png`

---

## How to Run

### Install Dependencies
```bash
pip install numpy matplotlib tensorflow keras
```

### Run Training
```bash
python src/cnn_fashion_mnist_epochs_summary.py
```

### Output Files Generated
- `sample_images.png` - Dataset sample visualization
- `epoch_vs_loss.png` - Loss progression graph
- `epoch_vs_accuracy.png` - Accuracy progression graph
---

## Summary

This project successfully demonstrates:
- ✓ Building a CNN from scratch using NumPy
- ✓ Understanding convolution, pooling, and fully-connected layers
- ✓ Forward and backward propagation implementation
- ✓ Training a neural network with SGD
- ✓ Achieving 89.24% training accuracy and 85.00% test accuracy on Fashion-MNIST
- ✓ Visualizing model performance with accuracy and loss graphs

The model shows strong learning capability with consistent improvements across epochs and good generalization from training to test data.
