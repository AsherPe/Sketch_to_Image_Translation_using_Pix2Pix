# Sketch-to-Image Translation Using Pix2Pix

Deep Learning project for translating grayscale flower sketches into realistic RGB flower images using Conditional GANs (Pix2Pix).

This project was developed as part of the **Advanced Topics in Deep Learning** course and focuses on image-to-image translation in the flower domain using the Oxford Flowers 102 dataset.

---

# Project Overview

The goal of this project is to learn a mapping from a sketch representation of a flower to a realistic RGB flower image.

Input:

- Grayscale sketch image (128×128×1)

Output:

- RGB flower image (128×128×3)

Two different approaches were implemented and compared:

1. Naive Encoder-Decoder baseline
2. Pix2Pix Conditional GAN

The project investigates how adversarial learning, PatchGAN discriminator size, and reconstruction loss weighting affect image quality and realism.

---

# Dataset

The project uses the **Oxford Flowers 102** dataset.

Each image was:

- Resized to 128×128 pixels
- Converted into a sketch representation
- Paired with its original RGB image

This produced supervised sketch-to-image training pairs.

### Data Normalization

Both sketches and target images were normalized to:

```text
[-1, 1]
```

to match the generator's final Tanh activation.

---

# Project Structure

```text
.
├── Flower Images
│   ├── Flower Image 1.png
│   └── Flower Image 2.png
├── Notebooks
│   ├── Experiments_Notebook.ipynb
│   ├── Inference Notebook.ipynb
│   └── Train Notebook.ipynb
├── README.md
└── Project_Report.pdf
```

## Train Notebook

Contains the complete training pipeline:

- Dataset loading
- Sketch generation
- Data preprocessing
- Naive baseline training
- Pix2Pix training
- PatchGAN comparison
- Model checkpointing
- Training visualization
- Final model selection

---

## Experiments Notebook

Contains additional experiments and analysis:

- Lambda (λ) comparison
- PatchGAN receptive field analysis
- Quantitative evaluation
- Visual comparison of generated images
- Hyperparameter experiments
- Model performance discussion

---

## Inference Notebook

Used for testing the final trained model.

Features:

- Load trained Pix2Pix generator
- Upload custom sketch images
- Generate flower images
- Visualize translation results

This notebook demonstrates the practical deployment of the trained model.

---

# Methodology

## Baseline Model

### Naive Encoder-Decoder

The baseline model consists of:

- Convolutional encoder
- Bottleneck representation
- Decoder with transposed convolutions

Training objective:

```text
Mean Squared Error (MSE)
```

### Limitations

The model tends to:

- Produce blurry outputs
- Lose fine texture information
- Average possible color distributions

This behavior is common when using only pixel-wise reconstruction losses.

---

## Pix2Pix Conditional GAN

The main model is based on the Pix2Pix framework.

Pix2Pix consists of two networks:

### Generator

- U-Net architecture
- Skip connections
- Tanh output activation

The skip connections preserve spatial information from the input sketch and help reconstruct detailed flower structures.

---

### Discriminator

PatchGAN discriminator

Instead of classifying an entire image as real or fake, PatchGAN evaluates local image patches.

This encourages:

- Sharper textures
- Better local realism
- More realistic flower details

---

## Generator Objective

The generator combines:

- Adversarial Loss
- L1 Reconstruction Loss

Objective:

```text
G* = argmin(G) max(D) LcGAN(G,D) + λLL1(G)
```

where λ controls the balance between realism and reconstruction fidelity.

---

# Training Configuration

| Parameter | Value |
|------------|------------|
| Image Size | 128 × 128 |
| Batch Size | 16 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Lambda Values | 50, 100, 200 |
| PatchGAN Variants | n=2, n=3, n=4 |
| Output Activation | Tanh |

---

# Experiments

## 1. Naive Baseline

The first experiment evaluated the encoder-decoder model trained using reconstruction loss only.

Purpose:

- Establish a baseline
- Compare adversarial vs non-adversarial training

---

## 2. PatchGAN Receptive Field Analysis

Different PatchGAN discriminator depths were evaluated:

```text
n = 2
n = 3
n = 4
```

### Observations

- n=2 produced unstable results
- n=3 generated sharp textures
- n=4 improved global consistency

This experiment demonstrates how discriminator receptive field size affects image quality.

---

## 3. Lambda (λ) Analysis

Different reconstruction weights were tested:

```text
λ = 50
λ = 100
λ = 200
```

### Observations

λ = 50

- Unrealistic colors
- Weak reconstruction

λ = 100

- Best balance between realism and fidelity

λ = 200

- Smoother outputs
- Less sharp details

---

# Results

## Quantitative Evaluation

| Model | MAE | MSE | PSNR | SSIM |
|---------|---------|---------|---------|---------|
| Naive Encoder-Decoder | 0.2879 | 0.1546 | 14.45 | 0.4344 |
| Pix2Pix (n=3, λ=100) | 0.2699 | 0.1459 | 14.89 | 0.6066 |
| Pix2Pix (n=4, λ=100) | 0.2670 | 0.1444 | 14.90 | 0.5999 |

---

# Final Model Selection

Although the Pix2Pix n=4 model achieved slightly better numerical metrics:

- Lower MAE
- Lower MSE
- Slightly higher PSNR

the Pix2Pix n=3 model achieved:

- Higher SSIM
- Sharper textures
- Better visual quality
- More realistic flower structures

Therefore, the final selected model was:

```text
Pix2Pix
PatchGAN n = 3
λ = 100
```

---

# Key Findings

- Pix2Pix significantly outperformed the naive encoder-decoder baseline.
- Adversarial training improved realism and texture quality.
- Skip connections helped preserve flower structure.
- PatchGAN discriminator size strongly influenced image appearance.
- λ = 100 provided the best balance between reconstruction and realism.
- Visual quality was a more reliable selection criterion than numerical metrics alone.

---

# Technologies Used

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- OpenCV
- Matplotlib
- Google Colab

---

# How to Run

## Training

Open:

```bash
Train Notebook.ipynb
```

Run all cells to:

- Load and preprocess the dataset
- Generate sketch-image pairs
- Train baseline and Pix2Pix models
- Save checkpoints
- Generate training visualizations

---

## Experiments

Open:

```bash
Experiments_Notebook.ipynb
```

Run all cells to reproduce:

- Lambda experiments
- PatchGAN comparison
- Evaluation metrics
- Visual analysis

---

## Inference

Open:

```bash
Inference Notebook.ipynb
```

Run all cells and upload a sketch image.

The notebook will:

- Load the trained generator
- Generate an RGB flower image
- Display the final result

---

# Sample Outputs

The repository contains visual comparisons between:

- Input sketch
- Naive baseline output
- Pix2Pix (n=3)
- Pix2Pix (n=4)
- Ground truth image

These comparisons highlight the improvement achieved by adversarial training.

---

# References

1. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A.

   **Image-to-Image Translation with Conditional Adversarial Networks**

   CVPR 2017.

2. Oxford Flowers 102 Dataset

   TensorFlow Datasets