# 下采样层,filters 为输出图层的通道数
# n * n * c -> 0.5n * 0.5n * filters
import keras
import tensorflow as tf

from PixelShuffler import PixelShuffler
import Setting


# 下采样层,filters 为输出图层的通道数
# n * n * c -> 0.5n * 0.5n * filters
def conv(filters):
    def block(x):
        x = keras.layers.convolutional.Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
        return x

    return block


# 上采样层，扩大图层大小
# 图层的形状变化如下：
# n*n*c -> n * n * 4filters -> 2n * 2n * filters
def upscale(filters):
    def block(x):
        x = keras.layers.convolutional.Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x

    return block


def Encoder():
    input_ = keras.layers.Input(shape=Setting.IMAGE_SHAPE)
    x = input_
    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(Setting.ENCODER_DIM)(x)
    x = keras.layers.Dense(4 * 4 * 1024)(x)
    x = keras.layers.Reshape((4, 4, 1024))(x)
    x = upscale(512)(x)
    return keras.models.Model(input_, x)


def Decoder():
    input_ = keras.layers.Input(shape=(8, 8, 512))
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return keras.models.Model(input_, x)


optimizer = tf.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()
x = keras.layers.Input(shape=Setting.IMAGE_SHAPE)
autoencoder_A = keras.models.Model(x, decoder_A(encoder(x)))
autoencoder_B = keras.models.Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')