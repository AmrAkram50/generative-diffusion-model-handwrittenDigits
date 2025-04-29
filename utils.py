import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pytorch_fid import fid_score

# Function to evaluate the generated images
def evaluate(model, dataloader):
    # Generate some images
    generated_images = model.generate_images(16)
    generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)

    # Save the generated images
    save_image(generated_images, 'generated_images.png', nrow=4)

    # Here we would compute MSE, PSNR, and FID (this is an example of calculating FID)
    fid = fid_score.calculate_fid_given_paths(['path_to_real_images', 'path_to_generated_images'], batch_size=64, cuda=True, dims=2048)
    print(f"FID score: {fid}")

# Function to save generated images
def save_images(model):
    generated_images = model.generate_images(16)
    generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)
    
    # Save images to disk
    save_image(generated_images, 'generated_digits.png', nrow=4)
    print("Images saved successfully!")
