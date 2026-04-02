import os
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def load_image(filepath):
    """Load an image and convert it to a NumPy array."""
    img = Image.open(filepath).convert('RGB')
    return np.array(img)

def compress_image_kmeans(image, n_colors):
    """
    Compress an image using k-means clustering.
    Args:
        image (ndarray): Original image as a NumPy array of shape (H, W, 3).
        n_colors (int): Number of colors to reduce the image to.
    Returns:
        ndarray: Compressed image.
    """
    # Reshape the image array to a 2D array of pixels (H*W, 3)
    h, w, c = image.shape
    pixels = image.reshape(-1, c)
    
    # Apply K-means clustering to find n_colors clusters
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Replace each pixel with its corresponding cluster center
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    
    # Reshape the compressed pixels back to the original image shape
    compressed_image = compressed_pixels.reshape(h, w, c)
    return compressed_image

def save_compressed_image(image, input_path, n_colors, quality=50):
    """
    Save the compressed image with optimized settings to reduce file size.
    Args:
        image (ndarray): Compressed image as a NumPy array.
        input_path (str): Path to the original input image.
        n_colors (int): Number of colors used for compression.
        quality (int): JPEG quality level (1-100, lower means higher compression).
    """
    # Get the directory and filename of the input image
    directory = os.path.dirname(input_path)
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create the output filename
    output_filename = f"{base_filename}kmeans{n_colors}colors_optimized.jpg"
    output_path = os.path.join(directory, output_filename)
    
    # Save the compressed image with optimized quality
    compressed_img = Image.fromarray(image)
    compressed_img.save(output_path, format='JPEG', quality=quality, optimize=True)
    print(f"Compressed image saved as: {output_path}")

if __name__ == "__main__":
    # Input file path
    input_filepath = input("Enter the path to the input image: ")  # User specifies input image path
    n_colors = int(input("Enter the number of colors to compress the image to: "))  # Number of colors
    quality = int(input("Enter the JPEG quality for saving (1-100, lower is more compressed): "))  # JPEG quality
    
    # Load the image
    original_image = load_image(input_filepath)
    
    # Compress the image using K-means
    compressed_image = compress_image_kmeans(original_image, n_colors)
    
    # Save the compressed image with optimized quality
    save_compressed_image(compressed_image, input_filepath, n_colors, quality)