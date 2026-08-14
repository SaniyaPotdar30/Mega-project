import cv2
import numpy as np

def preprocess_image(image, apply_grayscale=True, apply_blur=True, apply_contrast=True, apply_sharpen=True):
    """
    Applies a series of image processing techniques to enhance text readability for OCR.
    
    Parameters:
        image (numpy.ndarray): The input BGR image.
        apply_grayscale (bool): Convert image to grayscale.
        apply_blur (bool): Apply Gaussian blur to reduce high-frequency noise.
        apply_contrast (bool): Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        apply_sharpen (bool): Apply a sharpening filter kernel.
        
    Returns:
        numpy.ndarray: The preprocessed color image (BGR format, ready for CRAFT).
    """
    if image is None:
        raise ValueError("Input image is empty or invalid.")
        
    processed = image.copy()
    
    # 1. Grayscale Conversion
    if apply_grayscale:
        # Convert BGR to Grayscale
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (Noise Reduction)
    if apply_blur:
        # Kernel size (3,3) provides light smoothing without obliterating thin text strokes
        if len(processed.shape) == 3:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
        else:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
    # 3. Contrast Enhancement (CLAHE)
    if apply_contrast:
        # CLAHE is preferred over simple global equalization (equalizeHist) because it prevents over-amplification of noise
        if len(processed.shape) == 3:
            # If color image, apply to Y channel of YUV color space
            yuv = cv2.cvtColor(processed, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
            
    # 4. Sharpening Filter
    if apply_sharpen:
        # Use a laplacian-based sharpening kernel to make edges crisper
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        processed = cv2.filter2D(processed, -1, kernel)
        
    # CRAFT text detector expects a BGR 3-channel image.
    # If the processing left us with a 1-channel grayscale image, convert it back to BGR.
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
    return processed
