import cv2
import numpy as np
import torch
import re
# pyrefly: ignore [missing-import]
import easyocr
# pyrefly: ignore [missing-import]
import pytesseract

class OCREngine:
    """
    A robust OCR Engine supporting EasyOCR and Tesseract backends.
    Includes advanced bounding box sorting and Indian Rupee (₹) symbol detection/normalizer.
    """
    def __init__(self, engine="easyocr", languages=['en'], use_cuda=None):
        """
        Initializes the OCR Engine.
        
        Parameters:
            engine (str): 'easyocr' or 'tesseract'. Defaults to 'easyocr'.
            languages (list): List of languages to load (for EasyOCR). Defaults to ['en'].
            use_cuda (bool, optional): Force CUDA/GPU for EasyOCR. If None, auto-detects.
        """
        self.engine_name = engine.lower()
        self.languages = languages
        
        if use_cuda is None:
            self.cuda = torch.cuda.is_available()
        else:
            self.cuda = use_cuda
            
        self.easyocr_reader = None
        self.initialize_engine()

    def initialize_engine(self):
        """
        Initializes the active OCR backend and verifies libraries are loaded.
        """
        if self.engine_name == "easyocr":
            print(f"Initializing EasyOCR Reader (languages={self.languages}, CUDA={self.cuda})...")
            try:
                self.easyocr_reader = easyocr.Reader(self.languages, gpu=self.cuda)
                print("EasyOCR Engine initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize EasyOCR. Error: {e}")
                print("Falling back to Tesseract OCR if available.")
                self.engine_name = "tesseract"
                
        if self.engine_name == "tesseract":
            print("Configured to use Tesseract OCR.")
            # Verify pytesseract works
            try:
                tess_version = pytesseract.get_tesseract_version()
                print(f"Tesseract Engine ready (version {tess_version})")
            except Exception as e:
                print(f"WARNING: Tesseract binary not found or not in PATH: {e}")
                print("If running locally on Windows, make sure Tesseract is installed and added to environment variables.")

    def clean_text_result(self, text):
        """
        Cleans and normalizes extracted text.
        Preserves non-numeric text labels (e.g. ABC MART, Milk 1L, Date)
        and cleanly formats numeric price regions with the Rupee (₹) symbol.
        """
        if not text:
            return ""

        text_orig = text.strip()

        # 1. Check if text contains non-numeric words (e.g. "ABC MART", "Milk 1L", "Date: 10-07-2026")
        # If text contains standard words, dates, or non-price labels, preserve original text!
        has_words = bool(re.search(r'[a-zA-Z]{2,}', text_orig))
        
        # Exception: "Rs" or "RS" at start of text is a price indicator
        if has_words and not re.match(r'^(?:Rs\.?|RS\.?)\b', text_orig, re.IGNORECASE):
            return text_orig

        # 2. Standardize OCR punctuation typos in numeric prices
        text = text_orig.replace('_', '.')
        text = text.replace(' ', '')
        
        # Fix dot spacing like '787. 50' or '787_ 50'
        text = re.sub(r'(\d+)\s*[\._,]\s*(\d{2})$', r'\1.\2', text)
        
        # 3. Match explicit Rupee symbol representations (e.g. '?', 'Rs.', 'RS', '₹')
        symbol_match = re.match(r'^(?:[₹\?]|Rs\.?|RS\.?)\s*(\d+(?:\.\d{2})?)$', text, re.IGNORECASE)
        if symbol_match:
            return f"₹{symbol_match.group(1)}"
            
        # 4. Match numeric price pattern misreads:
        # e.g., '760.00' (was ₹60.00), '735.00' (was ₹35.00), '885.00' (was ₹85.00), '7120.00' (was ₹120.00), '7450.00' (was ₹450.00), '7750.00' (was ₹750.00), '737.50' (was ₹37.50)
        misread_price = re.match(r'^([78])(\d{2,3}\.\d{2})$', text)
        if misread_price:
            leading_digit, price_val = misread_price.groups()
            return f"₹{price_val}"

        # 5. If starts with '?' or '₹' directly
        if text.startswith('?') or text.startswith('₹'):
            cleaned = re.sub(r'^[^\d]*', '', text)
            return f"₹{cleaned}" if cleaned else text_orig

        return text_orig

    def sort_boxes(self, boxes):
        """
        Sorts bounding boxes spatially: top-to-bottom, then left-to-right.
        Groups boxes on the same horizontal line before sorting them left-to-right.
        
        Parameters:
            boxes (list): List of CRAFT boxes [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
            
        Returns:
            list: List of sorted box coordinates.
        """
        if boxes is None or len(boxes) == 0:
            return []
            
        # Convert 4-point coordinates to rectangular bounding boxes (startX, startY, width, height, original_box)
        processed_boxes = []
        for box in boxes:
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            
            startX = int(min(x_coords))
            startY = int(min(y_coords))
            endX = int(max(x_coords))
            endY = int(max(y_coords))
            
            width = endX - startX
            height = endY - startY
            
            # Keep original box coordinates alongside for spatial operations
            processed_boxes.append((startX, startY, width, height, box))
            
        # Step 1: Sort primarily by Y-coordinate (top-to-bottom)
        processed_boxes.sort(key=lambda b: b[1])
        
        sorted_boxes = []
        current_line = [processed_boxes[0]]
        
        # Step 2: Group boxes on the same logical line based on height overlap
        for b in processed_boxes[1:]:
            prev_y = current_line[-1][1]
            prev_h = current_line[-1][3]
            
            # If the current box's top boundary (startY) is close enough to the previous box's baseline
            # (within 50% of the previous box's height), group it on the same line.
            if abs(b[1] - prev_y) < (prev_h * 0.5):
                current_line.append(b)
            else:
                # Sort the accumulated line left-to-right (by X coordinate)
                current_line.sort(key=lambda l: l[0])
                sorted_boxes.extend(current_line)
                current_line = [b]
                
        # Append and sort the last line
        current_line.sort(key=lambda l: l[0])
        sorted_boxes.extend(current_line)
        
        # Return only the original 4-point box formats in their sorted order
        return [b[4] for b in sorted_boxes]

    def extract_text_from_region(self, cropped_image):
        """
        Runs the active OCR engine on a cropped region of an image.
        
        Parameters:
            cropped_image (numpy.ndarray): OpenCV sub-image.
            
        Returns:
            str: Extracted text.
        """
        if cropped_image is None or cropped_image.size == 0:
            return ""
            
        try:
            # Upscale small crops slightly for enhanced character recognition
            h, w = cropped_image.shape[:2]
            if h < 40 or w < 80:
                scale = 2.0
                target_w = int(w * scale)
                target_h = int(h * scale)
                target_crop = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            else:
                target_crop = cropped_image

            # Run active OCR engine on crop
            if self.engine_name == "easyocr":
                if self.easyocr_reader is None:
                    raise ValueError("EasyOCR is not initialized.")
                results = self.easyocr_reader.readtext(target_crop, detail=0)
                raw_text = " ".join(results).strip()
                
            elif self.engine_name == "tesseract":
                raw_text = pytesseract.image_to_string(target_crop, config="--psm 7").strip()
                
            else:
                raise ValueError(f"Unknown OCR engine configured: {self.engine_name}")
                
            # Clean and normalize extracted text
            return self.clean_text_result(raw_text)

        except Exception as e:
            print(f"Error extracting text from region: {e}")
            return ""

    def process_image_regions(self, original_image, boxes):
        """
        Crops, pre-processes, and extracts text from each detected bounding box region.
        
        Parameters:
            original_image (numpy.ndarray): The source image.
            boxes (list): The CRAFT bounding boxes to extract.
            
        Returns:
            tuple: (merged_text, regions_detail)
                   - merged_text (str): All texts combined logically.
                   - regions_detail (list): Dictionary details for each cropped box.
        """
        if original_image is None:
            return "", []
            
        if boxes is None or len(boxes) == 0:
            return "No text detected by model.", []
            
        # 1. Sort bounding boxes for natural logical layout reading
        sorted_boxes = self.sort_boxes(boxes)
        
        (H, W) = original_image.shape[:2]
        extracted_texts = []
        regions_detail = []
        
        # Add safety padding to box crops (stops characters at boundary edges from getting cut off)
        padding = 4
        
        for i, box in enumerate(sorted_boxes):
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            
            startX = int(min(x_coords))
            startY = int(min(y_coords))
            endX = int(max(x_coords))
            endY = int(max(y_coords))
            
            # Apply padding with border clamping
            startY_pad = max(0, startY - padding)
            endY_pad = min(H, endY + padding)
            startX_pad = max(0, startX - padding)
            endX_pad = min(W, endX + padding)
            
            # Crop region
            crop = original_image[startY_pad:endY_pad, startX_pad:endX_pad]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            # Perform OCR on the cropped box
            text = self.extract_text_from_region(crop)
            
            if text:
                extracted_texts.append(text)
                regions_detail.append({
                    "region_index": i + 1,
                    "box": [[int(p[0]), int(p[1])] for p in box],
                    "rectangle": [startX, startY, endX - startX, endY - startY],
                    "text": text
                })
                
        # Merge all elements separated by spaces or linebreaks
        merged_text = "\n".join(extracted_texts).strip()
        
        if not merged_text:
            merged_text = "No readable text extracted from regions."
            
        return merged_text, regions_detail
