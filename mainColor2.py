"""
使用K-means聚类出图片的k个主要颜色
请先压缩图像
太大的图像迭代起来贼慢
慢到脱发
"""

# img file
import random

import numpy as np
import warnings

warnings.filterwarnings('ignore')
# img_file = r'test.jpg'
# img_file = r'lilac_small_ex.jpg'
# img_file = r'ehookeri_small.jpg'
# img_file = r'mangolia_small.JPG'
# img_file = r'lucifer.JPG'
# img_file = r'Sinocalycanthus.JPG'
# img_file = r'Nopalxochia.JPG'
# img_file = r'Thalictrum.JPG'
# img_file = r'Papaver.JPG'
# img_file = r'Meconopsis.JPG'
# img_file = r'Corydalis.JPG'
# img_file = r'Rhodoleia.JPG'
# img_file = r'Amsonia.JPG'
# img_file = r'Roscoea.JPG'
# img_file = r'Corydalis caudata.JPG'
img_file = r'bus.jpg'

# k in k-means
k = 4


def get_img(file):
    """
    就是get图像啦
    :param file: 图片名
    :return: 图片的矩阵表示
    """
    from skimage import io
    return io.imread(file)


# get img
img = get_img(img_file)

# reshape 2D img to 1D
img_ori_shape = img.shape
img = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))
img_shape = img.shape
print('img size:', img_ori_shape)

# get pixels counts
n_pixels = img_shape[0]

# get channels count
n_channels = img_shape[1]

# init centers，随机选
centers = []
for i in range(k):
    random_pixel = random.randint(0, n_pixels - 1)
    centers.append(img[random_pixel])
centers = np.array(centers)
print('init centers:\n', centers)

# init labels
labels = np.zeros(n_pixels, dtype=int)

max_iter = 10


def get_euclidean_distance(_pixel, _center):
    """
    获得欧氏距离
    :param _pixel: 像素
    :param _center: 中心
    :return: 欧氏距离
    """
    d_pow_2 = 0
    for _channel_index in range(n_channels):
        d_pow_2 += pow(_pixel[_channel_index] - _center[_channel_index], 2)
    return np.sqrt(d_pow_2)


def get_nearest_center(_pixel):
    """
    获得最近的中心
    :param _pixel: 像素
    :return: 最近的中心的index
    """
    min_center_d = get_euclidean_distance(_pixel, centers[0])
    min_center_index = 0
    for _center_index in range(1, k):
        d = get_euclidean_distance(_pixel, centers[_center_index])
        if d < min_center_d:
            min_center_d = d
            min_center_index = _center_index
    return min_center_index


def cal_new_center():
    """
    计算新的中心
    如果有的中心没有最邻近的，那就说明选到了俩一样颜色的中心（虽然概率很小）那就重新选一个
    :return: 新的中心
    """

    # 归零
    center_counts = np.zeros(k, dtype=int)
    _centers = np.zeros((k, n_channels), dtype=int)

    # 对每个类逐通道求和
    for _pixel_index in range(n_pixels):
        center_counts[labels[_pixel_index]] += 1
        for _channel_index in range(n_channels):
            _centers[labels[_pixel_index]][_channel_index] += img[_pixel_index][_channel_index]

    # 对每个中心算出均值
    for _center_index in range(k):
        if center_counts[_center_index] > 0:
            for _channel_index in range(n_channels):
                _centers[_center_index][_channel_index] /= center_counts[_center_index]
        else:
            # 要是中心选重了就重新再选一个（虽然概率很小）
            _centers[_center_index] = img[random.randint(0, n_pixels - 1)]
            print('WARNING: Center %d has no pixel, re-choose center randomly...' % _center_index)

    return _centers


print('start iter')
# 迭代
for iter_index in range(max_iter):
    print('\niter %d...' % iter_index)
    changed_pixel = 0
    # 对每个像素计算属于哪个类别
    for pixel_index in range(n_pixels):
        label = get_nearest_center(img[pixel_index])
        if label != labels[pixel_index]:
            changed_pixel += 1
            labels[pixel_index] = label
    print('label', labels)
    # 如果少于1%的像素被聚到别的类中，停止迭代
    if changed_pixel / n_pixels < 0.01:
        break
    # 重新计算中心
    centers = cal_new_center()
    print(centers)

print()
print('\n=========================\nIter finished!')
print('Iter for %d iters' % iter_index)
print(centers)
print(labels)

# get result
# 获取centers对应的颜色值
# 并按照每个类别所包含的像素数量倒序排序
# 再输出成一个bmp
center_counts = {}
for label in labels:
    if label not in center_counts:
        center_counts[label] = 0
    center_counts[label] += 1
centers_index_sorted=[center[0] for center in sorted(center_counts.items(), key=lambda center: center[1], reverse=True)]

result = []
result_width = 200
result_height_per_center = 80
for center_index in centers_index_sorted:
    result.append(np.full((result_width * result_height_per_center, n_channels), centers[center_index], dtype=int))
result = np.array(result)
result = result.reshape((result_height_per_center * k, result_width, n_channels))


def save_img(_ori_file_name, _result):
    """
    保存成图片儿
    :param _ori_file_name: 源文件名
    :param _result: 获得的几个中心对应的颜色值凑出的矩阵
    :return: 木有
    """
    from skimage import io
    io.imsave(_ori_file_name.replace('.', '_result.'), _result)


save_img(img_file, result)
