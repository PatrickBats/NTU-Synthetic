from PIL import Image
import numpy as np

def categorize_size(image_path, white_threshold=250):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)  # Shape: (H, W, 3)

    # Create mask of white pixels: all RGB values above threshold
    white_mask = np.all(arr > white_threshold, axis=-1)

    # Invert to get non-white pixels
    nonwhite_mask = ~white_mask
    nonwhite_ratio = np.mean(nonwhite_mask)

    # Categorize based on ratio
    if nonwhite_ratio < 0.20:
        return "small"
    elif nonwhite_ratio <= 0.60:
        return "medium"
    else:
        return "large"

# Example usage:
image_path = "/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/ImageEditing/third_party/SizeEval/ImageSizers/base_small_smooth_02_tennisracquet.png"
size_category = categorize_size(image_path)
print(f"The object in the image is categorized as: {size_category}")
