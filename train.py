from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from models.gman import GMAN
from utils.dataset import DehazeDataset
from utils.metrics import psnr, ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets
train_data = DehazeDataset("data/train/hazy", "data/train/clean")
val_data   = DehazeDataset("data/val/hazy", "data/val/clean")

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=1, shuffle=False)

print("train loader: ", len(train_loader))
print("val loader: ", len(val_loader))

model = GMAN().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

print("STARTING TRAINING...")
for epoch in range(epochs):
    # -------- TRAIN --------
    print("Epoch {}/{}".format(epoch+1, epochs))
    model.train()
    train_loss = 0
    i = 0
    for hazy, clean in train_loader:
        print("i: ", i)
        i = i + 1
        hazy, clean = hazy.to(device), clean.to(device)

        out = model(hazy)
        loss = criterion(out, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"\nEpoch {epoch} Train Loss: {train_loss:.4f}")

    # -------- VALIDATION --------
    model.eval()
    total_psnr, total_ssim = 0, 0

    with torch.no_grad():
        j = 0
        for hazy, clean in val_loader:
            print("j: ", j)
            j = j + 1
            hazy, clean = hazy.to(device), clean.to(device)

            out = model(hazy)

            total_psnr += psnr(out, clean).item()
            total_ssim += ssim(out, clean).item()
            if(j == 1000):
                break

    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    print(f"Validation PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

# Save model
torch.save(model.state_dict(), "gman.pth")