import keras


# 子像素卷积操作
class PixelShuffler(keras.engine.base_layer.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = keras.backend.normalize_data_format(data_format)
        self.size = keras.utils.conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):
        input_shape = keras.backend.int_shape(inputs)
        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = self.size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = keras.backend.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = keras.backend.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
        out = keras.backend.reshape(out, (batch_size, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):
        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
        channels = input_shape[3] // self.size[0] // self.size[1]
        return input_shape[0], height, width, channels

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
