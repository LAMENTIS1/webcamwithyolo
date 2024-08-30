from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Read the image file from the request
    file = request.files['file'].read()
    image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Run inference with YOLOv8
    results = model(image)

    # Extract predictions
    predictions = results.pandas().xyxy[0].to_dict(orient='records')
    
    # Prepare results to send back
    response = {
        'predictions': predictions
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
