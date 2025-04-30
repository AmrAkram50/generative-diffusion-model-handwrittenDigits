import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
random_seed = 42
torch.manual_seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
denoise_model = DenoiseNet()
diffusion = DiffuseNet(model=denoise_model, num_steps=1000)
optimizer = optim.Adam(diffusion.model.parameters(), lr=1e-3)
epochs = 15
for epoch in range(1, epochs+1):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        optimizer.zero_grad()
        loss = diffusion.compute_loss(real_imgs)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
