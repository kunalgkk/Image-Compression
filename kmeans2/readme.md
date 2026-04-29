# Image Compression using K-Means

Compress images with K-Means color quantization while preserving quality.

## Install

pip install -r requirements.txt

## Run

Single compression:

python main.py --input input/sample.jpg --clusters 32

Higher quality:

python main.py --input input/sample.jpg --clusters 128

Batch compare:

python main.py --input input/sample.jpg --batch


## How it works

- Reads image pixels
- Runs K-Means clustering
- Reduces millions of colors to K representative colors
- Reconstructs image
- Saves optimized image
- Computes:
  - PSNR
  - SSIM
  - MSE
  - Compression Ratio

## Typical clusters

16   maximum compression  
32   balanced  
64   high quality  
128 nearly lossless
