import torch
import torch.nn.functional as F

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

def ssim(img1, img2):
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1**2
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    return (((2*mu1*mu2 + C1)*(2*sigma12 + C2)) /
           ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))).mean()