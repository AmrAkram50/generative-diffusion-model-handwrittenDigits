import torch
import torch.nn as nn

# U-Net based architecture for denoising
class DenoiseNet(nn.Module):
    def __init__(self, input_channels=2, base_channels=64):
        super(DenoiseNet, self).__init__()
        # Downsampling layers
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)
        # Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        d1 = torch.relu(self.conv1(x))
        d2 = torch.relu(self.conv2(d1))
        d3 = torch.relu(self.conv3(d2))
        u1 = torch.relu(self.deconv1(d3)) + d2
        u2 = torch.relu(self.deconv2(u1)) + d1
        return self.conv_out(u2)

# Diffusion process definition
class DiffuseNet:
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.num_steps = num_steps
        self.beta_schedule = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_prod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_prod = torch.sqrt(self.alpha_prod)
        self.sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alpha_prod)

    def compute_loss(self, x0):
        batch_size = x0.size(0)
        t = torch.randint(0, self.num_steps, (batch_size,))
        noise = torch.randn_like(x0)
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        sqrt_alpha_prod_t = self.sqrt_alpha_prod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alpha_prod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_prod_t * x0 + sqrt_one_minus_alpha_prod_t * noise
        t_norm = t.float() / self.num_steps
        t_channel = t_norm.view(-1, 1, 1, 1).repeat(1, 1, x0.shape[2], x0.shape[3])
        pred_noise = self.model(torch.cat([x_t, t_channel], dim=1))
        return torch.mean((noise - pred_noise) ** 2)

    def generate_images(self, num_samples):
        self.model.eval()
        x = torch.randn(num_samples, 1, 28, 28)
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long)
            t_channel = (t_tensor.float() / self.num_steps).view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
            pred_noise = self.model(torch.cat([x, t_channel], dim=1))
            x_prev_mean = (1 / self.alpha[t].sqrt()) * (x - (1 - self.alpha[t]) / (1 - self.alpha_prod[t]).sqrt() * pred_noise)
            if t > 0:
                z = torch.randn_like(x)
                x = x_prev_mean + self.beta_schedule[t].sqrt() * z
            else:
                x = x_prev_mean
        return x
