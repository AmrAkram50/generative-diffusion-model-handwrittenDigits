# Handwritten Digit Generation using Diffusion Models

This project implements **Denoising Diffusion Probabilistic Models (DDPMs)** to generate handwritten digits using the **MNIST dataset**. We will use a simplified version of the U-Net architecture to model the denoising process and train the model to generate digits similar to those in the MNIST dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Evaluation and Metrics](#evaluation-and-metrics)
5. [Code Structure](#code-structure)
6. [Future Work](#future-work)
7. [References](#references)

---

## Introduction

Generative modeling refers to creating new data that looks like the existing data. We use a **diffusion model** to generate handwritten digits. These models are known for producing high-quality outputs while being more stable during training than GANs.

To get started install requirements.txt
```shell script
!pip install -r requirements.txt
```

### Key Concepts
- **DDPM**: Adds noise to images step by step, then learns to reverse this process.
- **U-Net**: A neural network with an encoder-decoder structure, great for image generation.

---

## Model Architecture

The model uses a simplified **U-Net** with:
- **Encoder**: Compresses the image into features.
- **Decoder**: Rebuilds the image using those features.
- **Skip Connections**: Help recover lost details.
- **Loss Function**: Mean Squared Error between true and predicted noise.

### Model Code (Simplified)

```python
class DenoiseNet(nn.Module):
    ...
    def forward(self, x):
        ...
        return self.conv_out(u2)

class DiffuseNet:
    ...
    def compute_loss(self, x0):
        ...
        return torch.mean((noise - pred_noise) ** 2)
```

---

## Training Process

### Steps:
- **Normalize** MNIST images to [-1, 1].
- Use **MSE loss** to train.
- Optimize using **Adam**.
- Train for **15 epochs**.

```python
for epoch in range(epochs):
    for real_imgs, _ in train_loader:
        ...
        loss = diffusion_model.compute_loss(real_imgs)
        ...
```

---

## Evaluation and Metrics

We assess model performance using:

- **MSE**: Measures difference from real images.
- **PSNR**: Evaluates image clarity.
- **FID**: Compares real and generated distributions.

```python
def evaluate(model, dataloader):
    ...
    fid = fid_score.calculate_fid_given_paths(...)
    print(f"FID score: {fid}")
```

---

## Code Structure

- `main.py`: Run training and generation.
- `model.py`: Contains U-Net and diffusion logic.
- `utils.py`: Handles image saving and evaluation.

---

## Future Work

- **Higher-Resolution Datasets**: Try CIFAR-10 or CelebA.
- **Optimize Model**: Improve speed and memory use.
- **Class Conditioning**: Generate specific digits on demand.

---

## References

- Ho et al., 2020 - [DDPM Paper (NeurIPS)](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
- Wikipedia - [Fréchet Inception Distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
- Wikipedia - [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
