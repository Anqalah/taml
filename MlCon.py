import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPool2D, Input, Flatten
import os
import numpy as np
from l1dist import L1Dist


class proccess(object):
    def load_model(self, model_file_name):
        if not os.path.exists(model_file_name):
            raise FileNotFoundError(f"Model file not found at {model_file_name}")
        
        self.model = tf.keras.models.load_model(
            model_file_name,
            custom_objects={
                'L1Dist': L1Dist,
                'BinaryCrossentropy': tf.losses.BinaryCrossentropy()
            }
        )
        return self.model

    def preprocess(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at {file_path}")
        
        img = tf.io.decode_jpeg(tf.io.read_file(file_path))
        img = tf.image.resize(img, (105, 105)) / 255.0
        return img


class Predict(object):
    def verify(self, model, detection_threshold, verification_treshold, threshold_verify, input_varify, validation_verify):
        results = []
        processor = proccess()

        for image in os.listdir(os.path.join('Aplication_data', 'Verification_Images')):
            input_img = processor.preprocess(os.path.join('Aplication_data', 'Input_images', 'input_image.jpg'))
            validation_img = processor.preprocess(os.path.join('Aplication_data', 'Verification_Images', image))

            result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(results)
        verified = verification > verification_treshold

        return verified, results
