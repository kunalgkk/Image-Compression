import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def mse(original, compressed):
    return np.mean(
        (original.astype(float) -
         compressed.astype(float)) **2
    )


def psnr(original, compressed):
    return peak_signal_noise_ratio(
        original,
        compressed,
        data_range=255
    )


def ssim_score(original, compressed):

    if len(original.shape)==3:
        return structural_similarity(
            original,
            compressed,
            channel_axis=2,
            data_range=255
        )

    return structural_similarity(
        original,
        compressed,
        data_range=255
    )


def compression_ratio(original_size,new_size):
    return original_size/new_size


def report(original,
           compressed,
           original_size,
           compressed_size):

    return {
        "MSE": mse(original,compressed),
        "PSNR": psnr(original,compressed),
        "SSIM": ssim_score(original,compressed),
        "Compression Ratio":
            compression_ratio(
                original_size,
                compressed_size
            )
    }
