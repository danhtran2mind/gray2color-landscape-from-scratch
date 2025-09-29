import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization,
    LayerNormalization, Dropout, MultiHeadAttention, Add, Reshape, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
import glob
import os
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# Disable mixed precision
tf.keras.mixed_precision.set_global_policy('float32')

@tf.keras.utils.register_keras_serializable()
class SelfAttentionLayer(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ln = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        batch_size, height, width, channels = input_shape
        attention_shape = (batch_size, height * width, channels)
        self.mha.build(query_shape=attention_shape, value_shape=attention_shape)
        self.ln.build(input_shape)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        attention_input = tf.reshape(x, [b, h * w, c])
        attention_output = self.mha(attention_input, attention_input)
        attention_output = tf.cast(attention_output, dtype=x.dtype)
        attention_output = tf.reshape(attention_output, [b, h, w, c])
        return self.ln(x + attention_output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

def attention_unet_model(input_shape=(1024, 1024, 1)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = SelfAttentionLayer(num_heads=2, key_dim=32)(c4)
    
    def attention_gate(g, s, num_filters):
        g_conv = Conv2D(num_filters, (1, 1), padding='same')(g)
        s_conv = Conv2D(num_filters, (1, 1), padding='same')(s)
        attn = tf.keras.layers.add([g_conv, s_conv])
        attn = tf.keras.layers.Activation('relu')(attn)
        attn = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attn)
        return s * attn
    
    u5 = UpSampling2D((2, 2))(c4)
    a5 = attention_gate(u5, c3, 64)
    u5 = Concatenate()([u5, a5])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    a6 = attention_gate(u6, c2, 32)
    u6 = Concatenate()([u6, a6])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    a7 = attention_gate(u7, c1, 16)
    u7 = Concatenate()([u7, a7])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    
    outputs = Conv2D(2, (1, 1), activation='tanh', padding='same')(c7)
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    HEIGHT, WIDTH = 1024, 1024
    model = attention_unet_model(input_shape=(HEIGHT, WIDTH, 1))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=7e-5), loss=tf.keras.losses.MeanSquaredError())