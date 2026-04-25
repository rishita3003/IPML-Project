Project Title: Single Image Dehazing using GMAN (Model-Agnostic CNN)

---

1. Dependencies / Software Requirements

---

* Python 3.8 or higher
* PyTorch
* Torchvision
* NumPy
* Pillow (PIL)
* Matplotlib (optional, for plotting)

Install dependencies using:

pip install -r requirements.txt

Hardware Requirements:

* CPU is sufficient
* GPU (CUDA) recommended for faster training

---

2. How to Run the Code

---

Step 1: Prepare Dataset

Organize dataset in the following structure:
Link to the datasets:
SOTS : https://utexas.app.box.com/s/uqvnbfo68kns1210z5k5j17cvazavcd1 
ITS : https://utexas.app.box.com/s/07nvat6vwuecn93zuvbsa4pcmptfk7m4/folder/130076035860 

data/
├── train/
│   ├── hazy/
│   └── clean/
├── val/
│   ├── hazy/
│   └── clean/
├── test/
│   ├── hazy/
│   └── clean/

* Training uses ITS dataset (subset)
* Testing uses SOTS dataset

---

Step 2: Train the Model

Run:

python train.py

Output:

* Trained model will be saved as: gman.pth

---

Step 3: Test the Model (Inference)

Run:

python test.py

Output:

* Dehazed images saved in: outputs/
* Average PSNR and SSIM printed in terminal

---

3. Brief Description of Files

---

models/

* gman.py:
  Contains implementation of the GMAN architecture including
  encoder-decoder structure and residual blocks.

utils/

* dataset.py:
  Custom dataset loader for paired hazy and clean images.
* metrics.py:
  Implements evaluation metrics such as PSNR and SSIM.

train.py:

* Trains the GMAN model using MSE loss and Adam optimizer.
* Performs validation using PSNR and SSIM.

test.py:

* Loads trained model and performs inference on test dataset.
* Saves output images and computes average PSNR and SSIM.

requirements.txt:

* Lists all required Python packages.

data/:

* Contains training, validation, and test datasets.

outputs/:

* Stores dehazed output images generated during testing.

README.txt:

* Instructions for running the project.

---

4. Notes

---

* This is a simplified implementation of GMAN.
* A subset of the RESIDE dataset is used due to computational constraints.
* Results may differ from the original paper.

---

5. References

---

Z. Liu et al., "Single Image Dehazing with a Generic Model-Agnostic
Convolutional Neural Network," IEEE Signal Processing Letters, 2019.
