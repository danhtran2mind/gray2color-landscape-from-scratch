import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add, Conv2D, Reshape, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.keras.mixed_precision.set_global_policy('float32')

def transformer_model(input_shape=(1024, 1024, 1), patch_size=8, d_model=32, num_heads=4, ff_dim=64, num_layers=2, dropout_rate=0.1):
    HEIGHT, WIDTH, _ = input_shape
    num_patches = (HEIGHT // patch_size) * (WIDTH // patch_size)
    
    inputs = Input(shape=input_shape)
    x = Conv2D(d_model, (patch_size, patch_size), strides=(patch_size, patch_size), padding='valid')(inputs)
    x = Reshape((num_patches, d_model))(x)
    
    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    x = Dense(2)(x)
    x = Reshape((HEIGHT // patch_size, WIDTH // patch_size, 2))(x)
    x = UpSampling2D(size=(patch_size, patch_size), interpolation='bilinear')(x)
    outputs = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    return Model(inputs, outputs)

if __name__ == "__main__":
    HEIGHT, WIDTH = 1024, 1024
    model = transformer_model(input_shape=(HEIGHT, WIDTH, 1), patch_size=8, d_model=32, num_heads=4, ff_dim=64, num_layers=2)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=7e-5), loss=tf.keras.losses.MeanSquaredError())