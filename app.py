from flask import Flask, jsonify, request
import os
import tensorflow as tf
import logging
from MlCon import proccess, Predict
from l1dist import L1Dist
from flask_cors import CORS
import gdown

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173","https://tafe-pi.vercel.app/"])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi model
MODEL_DIR = 'model'
MODEL_FILE = 'siamesemodel_FIX.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1Xrx2tVFimDhWn9aWUCjrwLKMtFvryXp5"

# Global model
model = None
predictor = None

def download_model():
    """Download model dari Google Drive jika belum ada"""
    try:
        # Buat direktori model jika belum ada
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Cek apakah model sudah ada
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading model from Google Drive...")
            gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
            logger.info(f"Model downloaded to {MODEL_PATH}")
        else:
            logger.info("Model already exists, skipping download")
            
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False
    
def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info("Model not found, downloading...")
            gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, fuzzy=True, quiet=False)
        logger.info("Loading model...")
        model_handler = proccess()
        model = model_handler.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise

# Load model saat aplikasi start
try:
    if download_model():
        load_model()
        predictor = Predict()
    else:
        raise RuntimeError("Model download failed")
except Exception as e:
    logger.critical(f"Application failed to start: {str(e)}")
    exit(1)

# Direktori untuk gambar referensi
REFERENCE_IMAGES_DIR = 'reference_images'

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Get form data
        student_id = request.form.get('studentId')
        input_file = request.files.get('image')

        if not student_id:
            return jsonify({"error": "Missing studentId"}), 400
        if not input_file:
            return jsonify({"error": "Missing image file"}), 400
        
        # Path to reference image
        ref_path = os.path.join(REFERENCE_IMAGES_DIR, f'{student_id}.jpg')
        if not os.path.exists(ref_path):
            return jsonify({"error": f"Reference image not found for {student_id}"}), 404

        # Read image bytes
        input_bytes = input_file.read()
        with open(ref_path, 'rb') as f:
            reference_bytes = f.read()

        # Verify face
        verified, confidence = predictor.verify(
            model=model,
            input_bytes=input_bytes,
            reference_bytes=reference_bytes,
            detection_threshold=0.5,
            verification_threshold=0.5
        )
        
        return jsonify({
            "verified": verified,
            "confidence": confidence
        })

    except Exception as e:
        logger.exception("Error during verification")
        return jsonify({"error": str(e)}), 500

# Endpoint untuk menyimpan gambar referensi baru
@app.route('/save-reference', methods=['POST'])
def save_reference():
    try:
        student_id = request.form.get('studentId')
        input_file = request.files.get('image')
        
        if not student_id or not input_file:
            return jsonify({"error": "Missing parameters"}), 400
            
        # Buat folder jika belum ada
        if not os.path.exists(REFERENCE_IMAGES_DIR):
            os.makedirs(REFERENCE_IMAGES_DIR)
            
        # Simpan gambar
        ref_path = os.path.join(REFERENCE_IMAGES_DIR, f'{student_id}.jpg')
        input_file.save(ref_path)
        
        return jsonify({
            "status": "success",
            "message": f"Reference image saved for {student_id}"
        })
        
    except Exception as e:
        logger.exception("Error saving reference image")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create reference_images directory if not exists
    if not os.path.exists(REFERENCE_IMAGES_DIR):
        os.makedirs(REFERENCE_IMAGES_DIR)
    app.run(host='0.0.0.0', port=5000, debug=True)
