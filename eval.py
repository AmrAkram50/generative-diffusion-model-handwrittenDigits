import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pytorch_fid import fid_score
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
gen_imgs = generated_images.detach().cpu()
gen_imgs = ((gen_imgs + 1) / 2).clamp(0, 1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
test_ds     = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=gen_imgs.size(0), shuffle=False,
                         num_workers=2, pin_memory=True)

real_batch, _ = next(iter(test_loader))
real_imgs = real_batch.cpu()
real_imgs = ((real_imgs + 1) / 2).clamp(0, 1)
mse  = F.mse_loss(gen_imgs, real_imgs).item()
psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
print(f"MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
for folder in ('fid_real', 'fid_gen'):
    if os.path.exists(folder):
        # clear existing files
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)
for i, img in enumerate(real_imgs):
    torchvision.utils.save_image(img, f'fid_real/{i}.png')
for i, img in enumerate(gen_imgs):
    torchvision.utils.save_image(img, f'fid_gen/{i}.png')
fid_score = calculate_fid_given_paths(
    ['fid_real', 'fid_gen'],
    batch_size=8,
    device='cpu',
    dims=2048
)
print(f"FID score: {fid_score:.2f}")
