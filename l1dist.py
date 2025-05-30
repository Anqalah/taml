import tensorflow as tf
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    """
    Custom layer to compute L1 distance between two embeddings.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, input_embedding, validation_embedding):
        return tf.abs(input_embedding - validation_embedding)