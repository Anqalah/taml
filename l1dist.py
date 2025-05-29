import tensorflow as tf
from flask import Flask, request, jsonify, make_response
import joblib as jb
import base64
import numpy as np
import os
from flask_cors import CORS

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from keras.preprocessing.image import img_to_array, load_img

# siamese L1 distance layer
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # simmilarity calculation - L1 distance
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)