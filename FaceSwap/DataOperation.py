import cv2
import numpy as np
import os

import Setting


def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png")]


def load_images(images):
    global all_images
    iter_all_images = (cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1) for img in images)
    for i, image in enumerate(iter_all_images).__iter__():
        if i == 0:
            all_images = np.empty((len(images),) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_image(images):
    images_shape = np.array(images.shape)
    # new_axes 得到的是三个列表。[0,2],[1,3],[4]
    # 告诉调用者新集合中的每个维度由旧集合中的哪些维度构成
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(images, axes=np.concatenate(new_axes)).reshape(new_shape)


# 数据增强
def random_transform(image):
    h, w = image.shape[0:2]
    # 随机初始化旋转角度，范围 -10 ~ 10 之间。
    rotation = np.random.uniform(-10, 10)
    # 随机初始化缩放比例，范围 0.95 ~ 1.05 之间。
    scale = np.random.uniform(0.95, 1.05)
    # 随机定义平移距离，平移距离的范围为 -0.05 ~ 0.05。
    tx = np.random.uniform(-0.05, 0.05) * w
    ty = np.random.uniform(-0.05, 0.05) * h
    # 定义放射变化矩阵，用于将之前那些变化参数整合起来。
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    # 进行放射变化，根据变化矩阵中的变化参数，将图片一步步的进行变化，并返回变化后的图片。
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 图片有 40% 的可能性被翻转
    if np.random.random() < 0.4:
        result = result[:, ::-1]
    return result


def random_warp(image):
    # 先设置映射矩阵
    assert image.shape == (256, 256, 3)
    # 设置 range_ = [ 48.,  88., 128., 168., 208.]
    range_ = np.linspace(128 - 80, 128 + 80, 5)
    mapx = np.broadcast_to(range_, (5, 5))  # 利用 Python 广播的特性将 range_ 复制 5 份。
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)
    # 将大小为 5*5 的map放大为 80*80 ，再进行切片，得到 64 * 64 的 map
    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    # 通过映射矩阵进行剪切和卷曲的操作，最后获得 64*64 的训练集图片
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    # 下面四行代码涉及到 target 的制作，该段代码会在下面进行阐述
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]  # umeyama 函数的定义见下面代码块
    target_image = cv2.warpAffine(image, mat, (64, 64))

    return warped_image, target_image


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38). 下面的Eq 都分别对应着论文中的公式
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image)
        wrap_img, target_img = random_warp(image)
        if i == 0:
            wrapped_images = np.empty((batch_size,) + wrap_img.shape, wrap_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, target_img.dtype)
        wrapped_images[i] = wrap_img
        target_images[i] = target_img
    return wrapped_images, target_images


images_A_paths = get_image_paths(Setting.IMAGE_PATH_A)
images_B_paths = get_image_paths(Setting.IMAGE_PATH_B)

images_A = load_images(images_A_paths) / 255.0
images_B = load_images(images_B_paths) / 255.0
images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))


def get_training_data_A():
    return get_training_data(images_A, Setting.batch_size)


def get_training_data_B():
    return get_training_data(images_B, Setting.batch_size)
