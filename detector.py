import os
import torch
import numpy as np
from craft_text_detector import Craft

class CraftDetector:
    """
    A robust wrapper class for the CRAFT (Character Region Awareness for Text Detection) model.
    It automatically configures CUDA support and refiner status, executing text detection entirely in memory.
    """
    def __init__(self, use_cuda=None, enable_refiner=True, text_threshold=0.4, link_threshold=0.4, low_text=0.25):
        """
        Initializes the CRAFT detector.
        
        Parameters:
            use_cuda (bool, optional): Force CUDA usage. If None, automatically detects CUDA capability.
            enable_refiner (bool): Use RefineNet to link character/word boxes into complete sentence lines.
        """
        # 1. Device Selection (CUDA vs CPU)
        if use_cuda is None:
            self.cuda = torch.cuda.is_available()
        else:
            self.cuda = use_cuda
            
        self.enable_refiner = enable_refiner
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.craft = None
        
        # Load the models lazily or on instantiation. Instantiating now prepares the API for instant inference.
        self.initialize_model()

    def initialize_model(self):
        """
        Initializes and loads the CRAFT and Link Refiner networks.
        """
        try:
            print(f"Loading CRAFT text detector model (CUDA={self.cuda}, Refiner={self.enable_refiner}, Text Threshold={self.text_threshold}, Low Text={self.low_text})...")
            # We set output_dir=None to prevent the library from writing outputs (cropped boxes, heatmaps) to disk.
            # This is critical for keeping server memory clean and avoiding file system bloat in Docker.
            self.craft = Craft(
                output_dir=None, 
                crop_type="box", 
                cuda=self.cuda, 
                refiner=self.enable_refiner,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                low_text=self.low_text
            )
            print("CRAFT models loaded successfully!")
        except Exception as e:
            print(f"Error loading CRAFT models: {e}")
            raise

    def detect_text(self, image):
        """
        Detects text regions inside a numpy BGR image.
        
        Parameters:
            image (numpy.ndarray): OpenCV image in BGR format.
            
        Returns:
            list: A list of bounding boxes. Each box is represented as a list of 4 coordinates:
                  [[x_top_left, y_top_left], [x_top_right, y_top_right], 
                   [x_bottom_right, y_bottom_right], [x_bottom_left, y_bottom_left]]
        """
        if self.craft is None:
            raise RuntimeError("CRAFT detector is not initialized.")
            
        if image is None:
            raise ValueError("Input image is invalid.")

        try:
            # detect_text accepts a numpy array directly!
            # Since output_dir is None, it won't write any images to disk.
            prediction = self.craft.detect_text(image)
            
            # The result contains 'boxes' (4-point word coordinates) and 'polys' (polygon coordinates)
            # Boxes are structured as: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
            return prediction.get("boxes", [])
            
        except Exception as e:
            print(f"Error occurred during CRAFT text detection: {e}")
            return []

    def unload(self):
        """
        Unloads models from memory to free up RAM or GPU memory.
        """
        if self.craft is not None:
            try:
                self.craft.unload_craftnet_model()
                self.craft.unload_refinenet_model()
                print("CRAFT models unloaded from memory.")
            except Exception as e:
                print(f"Error unloading CRAFT models: {e}")
