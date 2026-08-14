import os
import time
import base64
import numpy as np
import cv2

from flask import Flask, request, jsonify
from flask_cors import CORS

# Hotpatch torchvision.models.vgg.model_urls before importing craft_text_detector (via detector)
# since it was removed in torchvision 0.15+
import torchvision.models.vgg
torchvision.models.vgg.model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}

# Hotpatch craft_text_detector.craft_utils.adjustResultCoordinates to support NumPy 1.24+
# by avoiding converting inhomogeneous list of arrays to np.array(polys) directly
import craft_text_detector.craft_utils as cu
def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = np.array(polys[k]) * (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
cu.adjustResultCoordinates = patched_adjustResultCoordinates
# Hotpatch craft_text_detector.predict.get_prediction to avoid ValueError on inhomogeneous arrays in NumPy 1.24+
import craft_text_detector.predict as predict_module
import craft_text_detector.image_utils as image_utils
import craft_text_detector.torch_utils as torch_utils
import craft_text_detector.craft_utils as craft_utils

def patched_get_prediction(
    image,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    long_size: int = 1280,
    poly: bool = True,
):
    t0 = time.time()
    image = image_utils.read_image(image)
    img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    x = image_utils.normalizeMeanVariance(img_resized)
    x = torch_utils.from_numpy(x).permute(2, 0, 1)
    x = torch_utils.Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()
    preprocessing_time = time.time() - t0
    t0 = time.time()

    with torch_utils.no_grad():
        y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch_utils.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()

    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    img_height = image.shape[0]
    img_width = image.shape[1]

    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    try:
        boxes_as_ratio = np.array(boxes_as_ratio)
    except ValueError:
        boxes_as_ratio = np.array(boxes_as_ratio, dtype=object)

    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])
    try:
        polys_as_ratio = np.array(polys_as_ratio)
    except ValueError:
        polys_as_ratio = np.array(polys_as_ratio, dtype=object)

    text_score_heatmap = image_utils.cvt2HeatmapImg(score_text)
    link_score_heatmap = image_utils.cvt2HeatmapImg(score_link)

    postprocess_time = time.time() - t0
    times = {
        "resize_time": resize_time,
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    return {
        "boxes": boxes,
        "boxes_as_ratios": boxes_as_ratio,
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "heatmaps": {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap,
        },
        "times": times,
    }

predict_module.get_prediction = patched_get_prediction
import craft_text_detector
craft_text_detector.get_prediction = patched_get_prediction

# Import our custom production modules
from preprocessor import preprocess_image
from detector import CraftDetector
from ocr import OCREngine

# ==========================================
# Flask App Setup
# ==========================================

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)  # Enable Cross-Origin Resource Sharing for API robustness

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ==========================================
# Initialize CRAFT and OCR Models on Startup
# ==========================================
# Models are loaded once during initialization and cached globally.
# This prevents reloading models on every API request, maximizing performance.

print("=" * 60)
print("STARTING PRODUCTION TEXT EXTRACTION API")
print("=" * 60)

try:
    # Use CUDA dynamically if available
    detector = CraftDetector(use_cuda=None, enable_refiner=True)
    # Default engine: EasyOCR (accurate, self-contained Python fallback).
    # Can also be set to 'tesseract' if Tesseract binary is installed.
    ocr_engine = OCREngine(engine="easyocr", languages=['en'], use_cuda=None)
except Exception as e:
    print(f"CRITICAL: Failed to load models during server startup: {e}")
    detector = None
    ocr_engine = None

print("=" * 60)
print("SERVER IS READY FOR REQUESTS")
print("=" * 60)

# ==========================================
# Home Route (Serves the HTML Dashboard)
# ==========================================

@app.route('/')
def home():
    return app.send_static_file('index.html')

# ==========================================
# Predict Route (CRAFT + OCR Text Extraction)
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    # --------------------------------------
    # 1. Check if image exists in request
    # --------------------------------------
    if 'image' not in request.files:
        return jsonify({"status": "error", "error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"status": "error", "error": "No image selected"}), 400

    try:
        # --------------------------------------
        # 2. Read uploaded image
        # --------------------------------------
        file_bytes = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_image is None:
            return jsonify({"status": "error", "error": "Invalid image format"}), 400

        # Check if backend models loaded correctly
        if detector is None or ocr_engine is None:
            return jsonify({
                "status": "error",
                "error": "Machine learning engines are not loaded on server.",
                "processing_time": f"{time.time() - start_time:.2f}s"
            }), 500

        (H, W) = original_image.shape[:2]

        # Get preprocessing parameters from requests (dynamic frontend tuning support)
        # Default all to true to run the complete enhancement suite
        apply_grayscale = request.form.get('grayscale', 'true').lower() == 'true'
        apply_blur = request.form.get('blur', 'true').lower() == 'true'
        apply_contrast = request.form.get('contrast', 'true').lower() == 'true'
        apply_sharpen = request.form.get('sharpen', 'true').lower() == 'true'

        # --------------------------------------
        # 3. Image Preprocessing
        # --------------------------------------
        preprocessed_image = preprocess_image(
            original_image,
            apply_grayscale=apply_grayscale,
            apply_blur=apply_blur,
            apply_contrast=apply_contrast,
            apply_sharpen=apply_sharpen
        )

        # --------------------------------------
        # 4. CRAFT Text Detection
        # --------------------------------------
        # We pass the preprocessed image for optimal edge activation
        boxes = detector.detect_text(preprocessed_image)
        # Save for offline debugging
        try:
            cv2.imwrite('/app/uploaded_test.png', original_image)
            cv2.imwrite('/app/preprocessed_test.png', preprocessed_image)
        except Exception as save_err:
            print(f"DEBUG: Failed to save test images: {save_err}", flush=True)
            
        print(f"DEBUG: Original shape: {original_image.shape}, Preprocessed shape: {preprocessed_image.shape}", flush=True)
        print(f"DEBUG: Detected boxes type: {type(boxes)}, count: {len(boxes) if boxes is not None else 0}", flush=True)

        # --------------------------------------
        # 5. Crop and OCR Text Extraction
        # --------------------------------------
        # We crop from the ORIGINAL image to preserve color details for the OCR reader
        merged_text, regions_detail = ocr_engine.process_image_regions(original_image, boxes)

        # --------------------------------------
        # 6. Visualizing Bounding Boxes & Text
        # --------------------------------------
        # Copy the original image to draw bounding boxes
        visualized_image = original_image.copy()
        
        # Draw bounding boxes for ALL detected CRAFT boxes
        for i, box in enumerate(boxes if boxes is not None else []):
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            # Bright neon green border (BGR: 118, 230, 0)
            cv2.polylines(visualized_image, [pts], isClosed=True, color=(118, 230, 0), thickness=2)

        # Draw text label badges for regions
        for region in regions_detail:
            box = region["box"]
            text = region["text"]
            
            # Find top-left coordinate for text labels
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            startX = max(0, int(min(x_coords)))
            startY = max(0, int(min(y_coords)))
            
            # Draw elegant label overlay if text is extracted
            if text.strip():
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                bubble_startY = max(text_h + 5, startY - 3)
                
                # Draw small neon green background for text label
                cv2.rectangle(
                    visualized_image,
                    (startX, bubble_startY - text_h - 4),
                    (startX + text_w + 6, bubble_startY + baseline - 2),
                    (118, 230, 0),
                    -1
                )
                
                # Draw black text inside green badge
                cv2.putText(
                    visualized_image,
                    text,
                    (startX + 3, bubble_startY - 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    lineType=cv2.LINE_AA
                )

        # --------------------------------------
        # 7. Convert Output Image to Base64
        # --------------------------------------
        _, buffer = cv2.imencode('.jpg', visualized_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # --------------------------------------
        # 8. Return JSON Response Contract
        # --------------------------------------
        end_time = time.time()
        processing_duration = f"{end_time - start_time:.2f}s"

        return jsonify({
            "status": "success",
            "detected_text": merged_text,
            "processing_time": processing_duration,
            "image_base64": encoded_image,
            "total_text_regions": len(regions_detail),
            "craft_boxes_count": len(boxes) if boxes is not None else 0,
            "regions": regions_detail,
            "hardware_accelerated": detector.cuda
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": f"Internal Server Error: {str(e)}"
        }), 500

# ==========================================
# Run Flask App
# ==========================================

if __name__ == '__main__':
    # Serve Flask on port 5000, open to host networks
    app.run(
        debug=False, # Debug is off because lazy model loaders run multiple threads in debug mode
        host='0.0.0.0',
        port=5000
    )