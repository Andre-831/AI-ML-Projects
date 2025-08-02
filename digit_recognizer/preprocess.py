import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Preprocess the image for model input.
    :param image_path: Path to the image file.
    :return: Preprocessed image as a numpy array.
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    # Reshape to match model input shape
    img = img.reshape(1, 28, 28, 1).astype('float32')
    
    return img