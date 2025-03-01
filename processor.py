import cv2
import numpy as np
from skimage import feature, color, morphology, transform
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_erosion, binary_dilation, binary_opening
import os
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, disk

def dice_similarity(A, B):
    intersection = np.logical_and(A, B)
    dice_coef = 2 * np.sum(intersection) / (np.sum(A) + np.sum(B))
    return dice_coef

def jaccard_coefficient(A, B):
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    jaccard_idx = np.sum(intersection) / np.sum(union)
    return jaccard_idx

def process_image(image_path):
    # Read and preprocess image
    rgb = cv2.imread(image_path)
    if rgb is None:
        raise ValueError("Could not read image")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float64) / 255.0

    # Convert to grayscale
    if len(rgb.shape) == 3:
        rgb1 = color.rgb2gray(rgb)
    else:
        rgb1 = rgb.astype(np.float64)

    # Create output directory
    output_dir = "static/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Apply Gaussian blur
    gb = gaussian_filter(rgb1, sigma=0.3)
    plt.imsave(os.path.join(output_dir, "1_gaussian_blur.png"), gb, cmap='gray')

    # Process image
    max1 = np.max(gb)
    max2 = max1 - 0.3
    loc = np.where(gb < max2)
    re = np.zeros_like(rgb1)
    re[loc] = rgb1[loc]

    # Edge detection
    lg = feature.canny(re)
    plt.imsave(os.path.join(output_dir, "2_edge_detection.png"), lg, cmap='gray')

    # Remove small objects
    rn1 = remove_small_objects(lg.astype(bool), min_size=1000)
    plt.imsave(os.path.join(output_dir, "3_cleaned_binary.png"), rn1, cmap='gray')

    # Morphological operations
    imd = binary_opening(rn1, structure=np.ones((3, 3)))
    selem_disk = disk(5)
    im2 = binary_dilation(imd, structure=selem_disk)
    plt.imsave(os.path.join(output_dir, "4_morphology.png"), im2, cmap='gray')

    # Final processing
    loc1 = np.where(rn1 == 1)
    re[loc1] = 0
    imd1 = binary_erosion(re, structure=np.ones((3, 3)), iterations=5)
    res = remove_small_objects(imd1.astype(bool), min_size=300)

    # Create colored overlay
    f = cv2.imread(image_path)
    if f is not None:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        m, n = rgb1.shape
        frs = np.zeros((m, n, 3), dtype=np.uint8)
        I1, J1 = np.where(res != 0)
        for jj in range(len(J1)):
            frs[I1[jj], J1[jj]] = f[I1[jj], J1[jj]]

        plt.imsave(os.path.join(output_dir, "5_final_result.png"), frs)

    # Calculate metrics
    dice_score = dice_similarity(lg, res)
    jaccard_score = jaccard_coefficient(lg, res)

    # Return results
    results = {
        'gaussian_blur': '1_gaussian_blur.png',
        'edge_detection': '2_edge_detection.png',
        'cleaned_binary': '3_cleaned_binary.png',
        'morphology': '4_morphology.png',
        'final_result': '5_final_result.png',
        'metrics': {
            'dice_coefficient': float(dice_score),
            'jaccard_index': float(jaccard_score),
            'detection_confidence': float(np.mean(res))
        }
    }

    return results