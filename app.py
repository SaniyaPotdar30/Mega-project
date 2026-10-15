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
from detector import CraftDetector, EastDetector
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
    # Initialize OpenCV DNN EAST text detector
    east_detector = EastDetector(model_path="frozen_east_text_detection.pb")
    # Default engine: EasyOCR (accurate, self-contained Python fallback).
    # Can also be set to 'tesseract' if Tesseract binary is installed.
    ocr_engine = OCREngine(engine="easyocr", languages=['en'], use_cuda=None)
except Exception as e:
    print(f"CRITICAL: Failed to load models during server startup: {e}")
    detector = None
    east_detector = None
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

@app.route('/health', methods=['GET'])
def health():
    return {"status": "healthy"}, 200
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
        # 4. Text Detection & OCR Pipeline Helper
        # --------------------------------------
        def process_model(model_name, det_boxes, box_color):
            if det_boxes is None:
                det_boxes = []
            
            # Crop & OCR extraction
            text_result, regions_detail = ocr_engine.process_image_regions(original_image, det_boxes)
            
            # Annotated visualization image
            vis_img = original_image.copy()
            for b in det_boxes:
                pts = np.array(b, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], isClosed=True, color=box_color, thickness=2)

            for region in regions_detail:
                box = region["box"]
                txt = region["text"]
                if txt.strip():
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    startX = max(0, int(min(x_coords)))
                    startY = max(0, int(min(y_coords)))
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
                    bubble_startY = max(th + 5, startY - 3)
                    
                    cv2.rectangle(
                        vis_img,
                        (startX, bubble_startY - th - 4),
                        (startX + tw + 6, bubble_startY + baseline - 2),
                        box_color,
                        -1
                    )
                    cv2.putText(
                        vis_img,
                        txt,
                        (startX + 3, bubble_startY - 2),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        lineType=cv2.LINE_AA
                    )
            
            _, buf = cv2.imencode('.jpg', vis_img)
            b64_img = base64.b64encode(buf).decode('utf-8')

            return {
                "detected_text": text_result,
                "image_base64": b64_img,
                "total_text_regions": len(regions_detail),
                "boxes_count": len(det_boxes),
                "regions": regions_detail
            }

        model_choice = request.form.get('model', 'both').lower()

        craft_res = None
        east_res = None

        if model_choice in ['craft', 'both']:
            craft_boxes = detector.detect_text(preprocessed_image) if detector is not None else []
            craft_res = process_model("CRAFT", craft_boxes, box_color=(118, 230, 0)) # Neon Green

        if model_choice in ['east', 'both'] and east_detector is not None:
            east_boxes = east_detector.detect_text(preprocessed_image)
            east_res = process_model("EAST", east_boxes, box_color=(0, 165, 255)) # Vibrant Orange

        end_time = time.time()
        processing_duration = f"{end_time - start_time:.2f}s"

        if model_choice == 'craft' or (model_choice == 'both' and craft_res and not east_res):
            primary_res = craft_res
        elif model_choice == 'east' and east_res:
            primary_res = east_res
        else:
            primary_res = craft_res if craft_res else east_res

        response_payload = {
            "status": "success",
            "model_used": model_choice.upper(),
            "processing_time": processing_duration,
            "detected_text": primary_res["detected_text"] if primary_res else "",
            "image_base64": primary_res["image_base64"] if primary_res else "",
            "total_text_regions": primary_res["total_text_regions"] if primary_res else 0,
            "craft_boxes_count": primary_res["boxes_count"] if primary_res else 0,
            "hardware_accelerated": detector.cuda if detector is not None else False
        }

        if craft_res:
            response_payload["craft"] = craft_res
        if east_res:
            response_payload["east"] = east_res

        return jsonify(response_payload), 200

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