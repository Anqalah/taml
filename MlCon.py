import tensorflow as tf
import numpy as np
import cv2
from l1dist import L1Dist  # Tambahkan impor ini

class proccess:
    def load_model(self, model_file_name):
        # Load model with custom objects
        self.model = tf.keras.models.load_model(
            model_file_name,
            custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy}
        )
        return self.model

    def preprocess(self, image_bytes):
        """Preprocess image directly from bytes"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (105, 105))  # Adjust to model input size
        img = img.astype(np.float32) / 255.0
        return img

class Predict:
    def verify(self, model, input_bytes, reference_bytes, detection_threshold=0.5, verification_threshold=0.5):
        """Verify face from image bytes"""
        processor = proccess()
        
        # Preprocess images
        input_img = processor.preprocess(input_bytes)
        reference_img = processor.preprocess(reference_bytes)
        
        # Expand dimensions for model input
        input_img = np.expand_dims(input_img, axis=0)
        reference_img = np.expand_dims(reference_img, axis=0)
        
        # Predict
        result = model.predict([input_img, reference_img])
        confidence = float(result[0][0])
        
        # Verification logic
        verified = confidence > verification_threshold
        
        return verified, confidence