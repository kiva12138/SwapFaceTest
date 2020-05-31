import numpy as np
import cv2
import DataOperation
import Setting
import matplotlib.pyplot as plt


images_A_paths = DataOperation.get_image_paths(Setting.IMAGE_PATH_A)
images_B_paths = DataOperation.get_image_paths(Setting.IMAGE_PATH_B)
print('Number of images_A is {}, number of images_B is {}'.format(len(images_A_paths), len(images_B_paths)))

A_Images = DataOperation.load_images(images_A_paths[:3])
B_Images = DataOperation.load_images(images_B_paths[:3])

figure = np.concatenate([A_Images, B_Images], axis=0)
print(figure.shape)
figure = figure.reshape((2, 3) + figure.shape[1:])
print(figure.shape)
figure = DataOperation.stack_image(figure)
print(figure.shape)
plt.imshow(cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
plt.show()
