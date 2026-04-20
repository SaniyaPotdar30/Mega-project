import os
import time
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify

# Serve index.html from the current folder
app = Flask(__name__, static_folder='.', static_url_path='')

# Path to the pretrained OpenCV EAST model
MODEL_PATH = "frozen_east_text_detection.pb"

# Try to load the model on startup
net = None
if os.path.exists(MODEL_PATH):
    print("Loading EAST text detector model...")
    net = cv2.dnn.readNet(MODEL_PATH)
else:
    print(f"WARNING: Model {MODEL_PATH} not found. Predictions will fail until downloaded.")

@app.route('/')
def home():
    """Serve the static root HTML file for the frontend."""
    return app.send_static_file('index.html')

# This is your main API

# Accepts image
# Returns result
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST request to detect text in an uploaded image.
    Expected form-data: 'image' containing an image file.
    """
    start_time = time.time()
    
    # 1. Error handling if no image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
    try:
        # 2. Read image from the upload byte stream
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Model check
        if net is None:
            return jsonify({
                "status": "error",
                "result": f"Model not found. Run download_model.py first.",
                "processing_time": f"{time.time() - start_time:.2f}s"
            })

        # 3. Process the image for the EAST model
        # The EAST model requires image dimensions to be a multiple of 32
        # We will resize to a standard 320x320 grid for speed and simplicity
        (H, W) = image.shape[:2]
        newW, newH = (320, 320)
        
        # Prepare the image as a blob
        blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        # 4. Pass the blob through the CV2 neural network
        net.setInput(blob)
        # We extract only the scores and geometry, required for EAST
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        
        # 5. Extract bounding boxes from EAST model output
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # Ignore low confidence scores
                if scoresData[x] < 0.5: 
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # Extract rotation angle and compute sine/cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # Map dimensions
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # Compute starting and ending coordinates
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(float(scoresData[x]))

        # Apply Non-Maxima Suppression to suppress overlapping boxes
        indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
        
        result_text = "No Text Detected"
        if len(indices) > 0:
            result_text = "Text Detected"
            # Calculate ratio to map boxes back to original image size
            rW = W / float(newW)
            rH = H / float(newH)
            
            for i in indices:
                # Handle varying array types from NMSBoxes based on OpenCV version
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                (startX, startY, endX, endY) = rects[idx]
                
                # Scale coordinates
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                
                # Draw the bounding box (green color, thickness 3)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)

        end_time = time.time()
        
        # Encode the processed image to send back to frontend
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        # Return structured JSON response matching requirements
        return jsonify({
            "status": "success",
            "result": result_text,
            "processing_time": f"{end_time - start_time:.2f}s",
            "image_base64": encoded_image
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start local testing server
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
