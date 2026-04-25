import torch
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

from models.gman import GMAN
from utils.metrics import psnr, ssim   # IMPORTANT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = GMAN().to(device)
model.load_state_dict(torch.load("gman.pth", map_location=device))
model.eval()

# Transform
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

# Paths
hazy_dir = "data/test/hazy"
clean_dir = "data/test/clean"
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

total_psnr = 0
total_ssim = 0
count = 0

# Loop through all test images
i = 0
for file in os.listdir(hazy_dir):
    print("image: ", i)
    i = i + 1
    hazy_path = os.path.join(hazy_dir, file)

    # Match clean image
    clean_name = file.split('_')[0] + ".png"
    clean_path = os.path.join(clean_dir, clean_name)

    # Load images
    hazy = Image.open(hazy_path).convert("RGB")
    clean = Image.open(clean_path).convert("RGB")

    hazy_tensor = transform(hazy).unsqueeze(0).to(device)
    clean_tensor = transform(clean).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(hazy_tensor)

    # Save output
    out_path = os.path.join(output_dir, file)
    vutils.save_image(output, out_path)

    # Compute metrics
    total_psnr += psnr(output, clean_tensor).item()
    total_ssim += ssim(output, clean_tensor).item()
    count += 1

    print(f"Processed: {file}")

# Final results
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print("\n===== FINAL RESULTS =====")
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")