import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Add, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.keras.mixed_precision.set_global_policy('float32')

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate()([avg_pool, max_pool])
        attention = self.conv(concat)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

def build_autoencoder(height, width):
    input_img = Input(shape=(height, width, 1))
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = SpatialAttention()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    residual = Conv2D(192, (1, 1), padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = SpatialAttention()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    residual = Conv2D(384, (1, 1), padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = SpatialAttention()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = SpatialAttention()(x)
    x = UpSampling2D((2, 2))(x)
    
    residual = Conv2D(192, (1, 1), padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = SpatialAttention()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialAttention()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(2, (3, 3), activation=None, padding='same')(x)
    
    return Model(input_img, decoded)

if __name__ == "__main__":
    HEIGHT, WIDTH = 512, 512
    autoencoder = build_autoencoder(HEIGHT, WIDTH)
    autoencoder.summary()
    autoencoder.compile(optimizer=Adam(learning_rate=7e-5), loss=tf.keras.losses.MeanSquaredError())