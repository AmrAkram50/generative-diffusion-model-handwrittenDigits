import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import DenoiseNet, DiffuseNet
from utils import evaluate, save_images

# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale pixel values to [-1, 1]
])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
denoise_model = DenoiseNet().to(device)
diffusion_model = DiffuseNet(model=denoise_model, num_steps=1000).to(device)
optimizer = optim.Adam(diffusion_model.model.parameters(), lr=1e-3)

# Training loop
epochs = 15
for epoch in range(epochs):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        optimizer.zero_grad()
        loss = diffusion_model.compute_loss(real_imgs)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# Evaluation and Image Generation
evaluate(diffusion_model, train_loader)
save_images(diffusion_model)
