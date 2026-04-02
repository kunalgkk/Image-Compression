import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import time
import psutil

def load_image(filepath):
    """Load an image and convert it to a NumPy array normalized to [0, 1]."""
    img = Image.open(filepath).convert('RGB')
    return np.array(img) / 255.0

def apply_pca_compression(image, n_components):
    """
    Apply PCA compression to each RGB channel separately.
    Args:
        image (ndarray): Image array of shape (H, W, 3).
        n_components (int): Number of principal components to retain.
    Returns:
        ndarray: Compressed and reconstructed image.
    """#K:\College\Smesters VIT-B\Interim semseter 2024-25\Assignments and certificates\Project exhibition\Codes\Project exbhition Picture.png
    compressed_channels = []
    for channel in range(3):  # Loop through R, G, B channels
        # Extract the channel
        img_channel = image[:, :, channel]
        # Apply PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(img_channel)
        reconstructed = pca.inverse_transform(transformed)
        compressed_channels.append(reconstructed)
    # Stack the channels back together
    return np.stack(compressed_channels, axis=-1)

def save_compressed_image(image, input_filepath, n_components):
    """
    Save the compressed image to a file in the same directory as the input image.
    Args:
        image (ndarray): Compressed image as a NumPy array.
        input_filepath (str): Path to the input image file.
        n_components (int): Number of components used in PCA compression.
    """
    # Get directory and base filename from input path
    input_dir = os.path.dirname(input_filepath)
    input_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    
    # Generate output file path
    output_filepath = os.path.join(input_dir, f"{input_filename}_pca_{n_components}_components.jpg")
    
    # Save the compressed image
    img_to_save = (np.clip(image, 0, 1) * 255).astype(np.uint8)  # Convert to 8-bit
    compressed_img = Image.fromarray(img_to_save)
    compressed_img.save(output_filepath)
    print(f"Compressed image saved as: {output_filepath}")

if __name__ == "__main__":
    # Input file path
    input_filepath = input("Enter the path to the input image: ")  # User specifies input image path
    
    # Load and normalize the image
    original_image = load_image(input_filepath)
    
    # Set the number of components to keep
    n_components = int(input("Enter the number of components to retain: "))  # Adjust for desired compression level

    # Start measuring execution time after input
    start_time = time.time()

    # Start measuring memory usage before applying PCA
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

    # Apply PCA compression
    compressed_image = apply_pca_compression(original_image, n_components)

    # Save the compressed image in the same directory
    save_compressed_image(compressed_image, input_filepath, n_components)

    # End measuring execution time
    end_time = time.time()

    # Calculate and display execution time (excluding input time)
    execution_time = end_time - start_time
    print(f"Execution time (excluding input): {execution_time:.2f} seconds")
    
    # Get memory usage after PCA processing
    memory_after = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    memory_used = memory_after - memory_before  # Memory used during PCA process
    print(f"Memory used by the program: {memory_used:.2f} MB")
