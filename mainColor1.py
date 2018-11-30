import numpy as np
from skimage import io
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

img_file = 'city.jpg'

# k in k-means
k = 5

# get img
img = io.imread(img_file)
# reshape 2D img to 1D
img_ori_shape = img.shape
img1 = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))
img_shape = img1.shape
print('img size:', img_ori_shape)

# get pixels counts
n_pixels = img_shape[0]

# get channels count
n_channels = img_shape[1]

estimator = KMeans(n_clusters=k, max_iter=4000, init='k-means++', n_init=50)  # 构造聚类器
estimator.fit(img1)  # 聚类
centroids = estimator.cluster_centers_  # 获取聚类中心


result = []
result_width = 200
result_height_per_center = 80
for center_index in range(k):
    result.append(np.full((result_width * result_height_per_center, n_channels), centroids[center_index], dtype=int))
result = np.array(result)
result = result.reshape((result_height_per_center * k, result_width, n_channels))

io.imsave(img_file + '.result.bmp', result)
