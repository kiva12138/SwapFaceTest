import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras

from PixelShuffler import PixelShuffler
import DataOperation

if __name__ == '__main__':
    autoencoder_A = keras.models.load_model('./modelA.h5', custom_objects={'PixelShuffler': PixelShuffler})
    autoencoder_B = keras.models.load_model('./modelB.h5', custom_objects={'PixelShuffler': PixelShuffler})

    warped_A, target_A = DataOperation.get_training_data_A()
    warped_B, target_B = DataOperation.get_training_data_B()

    test_A = target_A[0:3]
    test_B = target_B[0:3]

    # 进行拼接 原图 A - 解码器 A 生成的图 - 解码器 B 生成的图
    figure_A = np.stack([test_A, autoencoder_A.predict(test_A), autoencoder_B.predict(test_A), ], axis=1)
    # 进行拼接  原图 B - 解码器 B 生成的图 - 解码器 A 生成的图
    figure_B = np.stack([test_B, autoencoder_B.predict(test_B), autoencoder_A.predict(test_B), ], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((2, 3) + figure.shape[1:])
    figure = DataOperation.stack_image(figure)
    figure = np.clip(figure * 255, 0, 255).astype('uint8')

    plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))
    plt.show()
