from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

img_file = 'city.jpg'

# k-means中的k值，即选择几个中心点
k = 5

# 读图片
img = io.imread(img_file)
# 转换数据维度
img_ori_shape = img.shape
img1 = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))
img_shape = img1.shape

# 获取图片色彩层数
n_channels = img_shape[1]

estimator = KMeans(n_clusters=k, max_iter=4000, init='k-means++', n_init=50)  # 构造聚类器
estimator.fit(img1)  # 聚类
centroids = estimator.cluster_centers_  # 获取聚类中心


colorLabels = list(estimator.labels_)
colorInfo = {}
for center_index in range(k):
    colorRatio = colorLabels.count(center_index)/len(colorLabels)
    colorInfo[colorRatio] = centroids[center_index]

# 根据比例排序，从高至第低
colorInfo = [(k,colorInfo[k]) for k in sorted(colorInfo.keys(), reverse=True)] 
for color in colorInfo:
    print('比例：', color[0], '颜色：', color[1])
    
    
# 使用算法跑出的中心点，生成一个矩阵，为数据可视化做准备
result = []
result_width = 200
result_height_per_center = 80
for center_index in range(k):
    result.append(np.full((result_width * result_height_per_center, n_channels), colorInfo[center_index][1], dtype=int))
result = np.array(result)
result = result.reshape((result_height_per_center * k, result_width, n_channels))

# 保存图片
io.imsave(os.path.splitext(img_file)[0] + '_result.bmp', result)
