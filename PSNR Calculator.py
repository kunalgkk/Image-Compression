import cv2
import numpy as np

def calculate_psnr(original, compressed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # Means images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main():
    # Get the file paths from the user
    original_path = input("Enter the path to the original image: ")
    compressed_path = input("Enter the path to the compressed image: ")
    
    try:
        # Load the images
        original = cv2.imread(original_path, cv2.IMREAD_COLOR)
        compressed = cv2.imread(compressed_path, cv2.IMREAD_COLOR)
        
        if original is None:
            print("Error: Original image not found or invalid format.")
            return
        if compressed is None:
            print("Error: Compressed image not found or invalid format.")
            return

        # Ensure the images have the same dimensions
        if original.shape != compressed.shape:
            print("Error: Images do not have the same dimensions.")
            return

        # Calculate PSNR
        psnr_value = calculate_psnr(original, compressed)
        print(f"PSNR between the two images: {psnr_value:.2f} dB")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
