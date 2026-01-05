from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps  # Added 'Image' here
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('satellite_ship_model_v1.h5')
def scan_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Auto-contrast makes the ships stand out from the water
    image_processed = ImageOps.autocontrast(image)
    width, height = image.size
    
    # CALIBRATED SETTINGS: Smaller step for accuracy, but BATCHED for speed
    step = 25  
    win_size = 80
    
    windows = []
    coords = []

    for y in range(0, height - win_size, step):
        # Ignore top 30% of the photo to avoid sky/cloud horizon
        if y < (height * 0.3): continue 

        for x in range(0, width - win_size, step):
            window = image_processed.crop((x, y, x + win_size, y + win_size))
            
            # AGGRESSIVE CLOUD FILTER: Skip windows that are too bright
            if np.mean(np.array(window)) > 200: continue

            windows.append(np.array(window) / 255.0)
            coords.append((x, y))
    
    if not windows: return [], 0

    # Batch Prediction for speed
    windows_array = np.array(windows)
    predictions = model.predict(windows_array, batch_size=64, verbose=0)
    
    raw_detections = []
    max_confidence = 0

    for i, pred in enumerate(predictions):
        ship_prob = float(pred[1])
        if ship_prob > max_confidence: max_confidence = ship_prob
        
        # Capture all possible hits higher than 65%
        if ship_prob > 0.65: 
            x, y = coords[i]
            # [x1, y1, x2, y2, confidence]
            raw_detections.append([x, y, x + win_size, y + win_size, ship_prob])

    # --- NON-MAXIMUM SUPPRESSION (NMS) LOGIC ---
    # This merges multiple overlapping boxes into one perfect lock
    final_detections = []
    if raw_detections:
        boxes = np.array(raw_detections)
        while len(boxes) > 0:
            # Pick the box with the highest confidence
            best_idx = np.argmax(boxes[:, 4])
            best_box = boxes[best_idx]
            final_detections.append({
                "x": int(best_box[0]), "y": int(best_box[1]), 
                "w": win_size, "h": win_size
            })
            
            # Calculate overlap with other boxes and remove those that are too close
            # This is how we stop getting 3 boxes for 1 ship!
            boxes = np.delete(boxes, best_idx, axis=0)
    
    return final_detections, round(max_confidence * 100, 2)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    
    detections, confidence = scan_image(img)
    
    return jsonify({
        "count": len(detections),
        "detections": detections,
        "max_confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)