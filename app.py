from flask import Flask, jsonify, request
import os
import numpy as np
import tensorflow as tf
from MlCon import proccess, Predict
from l1dist import L1Dist

app = Flask(__name__)

# Load model saat aplikasi dijalankan
model_handler = proccess()
model = model_handler.load_model('model/siamesemodel_FIX.h5')  # Ganti dengan path model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        detection_threshold = data.get('detection_threshold', 0.5)
        verification_threshold = data.get('verification_threshold', 0.5)

        predictor = Predict()
        verified, results = predictor.verify(
            model=model,
            detection_threshold=detection_threshold,
            verification_treshold=verification_threshold,
            threshold_verify=None,
            input_varify=None,
            validation_verify=None
        )

        return jsonify({
            "verified": bool(verified),
            "results": [float(r[0][0]) for r in results]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
