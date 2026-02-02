"""
Effect Utilities

Reusable building blocks for composing effects.
These functions can be chained together to create complex effects.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# COLOR SMOOTHING
# =============================================================================

def bilateral_smooth(image: np.ndarray, iterations: int = 5, d: int = 9,
                     sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    """
    Smooth colors using iterative bilateral filtering.
    Preserves edges while smoothing flat regions.
    
    Args:
        image: Input BGR image
        iterations: Number of filter passes (more = smoother)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Smoothed color image
    """
    result = image.copy()
    for _ in range(iterations):
        result = cv2.bilateralFilter(result, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return result


# =============================================================================
# EDGE DETECTION
# =============================================================================

def laplacian_edges(image: np.ndarray, blur_ksize: int = 5, laplacian_ksize: int = 5,
                    threshold: int = 50, invert: bool = True) -> np.ndarray:
    """
    Detect edges using Laplacian operator.
    
    Args:
        image: Input BGR image
        blur_ksize: Gaussian blur kernel size (must be odd)
        laplacian_ksize: Laplacian kernel size (must be odd, 1-7)
        threshold: Threshold for edge binarization
        invert: If True, returns white background with black edges
    
    Returns:
        Binary edge mask (single channel)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=laplacian_ksize)
    
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    edges = cv2.threshold(edges, threshold, 255, thresh_type)[1]
    return edges


def adaptive_edges(image: np.ndarray, blur_ksize: int = 5, block_size: int = 9,
                   c: int = 2) -> np.ndarray:
    """
    Detect edges using adaptive thresholding.
    
    Args:
        image: Input BGR image
        blur_ksize: Median blur kernel size (must be odd)
        block_size: Size of pixel neighborhood for threshold (must be odd)
        c: Constant subtracted from mean
    
    Returns:
        Binary edge mask (single channel)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur_ksize)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block_size, c)
    return edges


def canny_edges(image: np.ndarray, blur_ksize: int = 5, low_threshold: int = 50,
                high_threshold: int = 150, invert: bool = True) -> np.ndarray:
    """
    Detect edges using Canny edge detector.
    
    Args:
        image: Input BGR image
        blur_ksize: Gaussian blur kernel size (must be odd)
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        invert: If True, returns white background with black edges
    
    Returns:
        Binary edge mask (single channel)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray_blur, low_threshold, high_threshold)
    
    if invert:
        edges = cv2.bitwise_not(edges)
    return edges


def sobel_edges(image: np.ndarray, blur_ksize: int = 5, sobel_ksize: int = 3,
                threshold: int = 50, invert: bool = True) -> np.ndarray:
    """
    Detect edges using Sobel operator (combined X and Y).
    
    Args:
        image: Input BGR image
        blur_ksize: Gaussian blur kernel size (must be odd)
        sobel_ksize: Sobel kernel size (1, 3, 5, or 7)
        threshold: Threshold for edge binarization
        invert: If True, returns white background with black edges
    
    Returns:
        Binary edge mask (single channel)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    sobel_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(255 * magnitude / magnitude.max())
    
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    edges = cv2.threshold(magnitude, threshold, 255, thresh_type)[1]
    return edges


# =============================================================================
# COMBINING / MASKING
# =============================================================================

def combine_with_edges(color_image: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """
    Combine a color image with an edge mask using bitwise AND.
    
    Args:
        color_image: BGR color image
        edge_mask: Single-channel binary mask (white = keep, black = edge lines)
    
    Returns:
        Combined cartoon-style image
    """
    return cv2.bitwise_and(color_image, color_image, mask=edge_mask)


def overlay_edges(color_image: np.ndarray, edge_mask: np.ndarray,
                  edge_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Overlay edges on top of a color image.
    
    Args:
        color_image: BGR color image
        edge_mask: Single-channel binary mask (white = edges)
        edge_color: BGR color for edges
    
    Returns:
        Image with edges overlaid
    """
    result = color_image.copy()
    result[edge_mask > 127] = edge_color
    return result


# =============================================================================
# COLOR QUANTIZATION
# =============================================================================

def quantize_colors(image: np.ndarray, num_colors: int = 8) -> np.ndarray:
    """
    Reduce the number of colors using K-means clustering.
    
    Args:
        image: Input BGR image
        num_colors: Number of colors to reduce to
    
    Returns:
        Color-quantized image
    """
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, 
                                     cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(image.shape)
    return quantized


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def ensure_odd(value: int, min_val: int = 1) -> int:
    """Ensure a value is odd (required by many OpenCV functions)."""
    value = max(min_val, int(value))
    return value if value % 2 == 1 else value + 1


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def grayscale_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to BGR."""
    if len(image.shape) == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


# =============================================================================
# COMPOSITE EFFECTS (Combining building blocks)
# =============================================================================

def cartoon_effect(image: np.ndarray, edge_type: str = "laplacian",
                   bilateral_iterations: int = 5, d: int = 9,
                   sigma_color: int = 75, sigma_space: int = 75,
                   **edge_params) -> np.ndarray:
    """
    Create cartoon effect by combining edge detection with color smoothing.
    
    Args:
        image: Input BGR image
        edge_type: "laplacian", "adaptive", "canny", or "sobel"
        bilateral_iterations: Number of bilateral filter passes
        d, sigma_color, sigma_space: Bilateral filter parameters
        **edge_params: Additional parameters for the chosen edge detector
    
    Returns:
        Cartoon-style image
    """
    # Step 1: Detect edges
    edge_funcs = {
        "laplacian": laplacian_edges,
        "adaptive": adaptive_edges,
        "canny": canny_edges,
        "sobel": sobel_edges
    }
    edge_func = edge_funcs.get(edge_type, laplacian_edges)
    edges = edge_func(image, **edge_params)
    
    # Step 2: Smooth colors
    color = bilateral_smooth(image, iterations=bilateral_iterations, d=d,
                             sigma_color=sigma_color, sigma_space=sigma_space)
    
    # Step 3: Combine
    return combine_with_edges(color, edges)
