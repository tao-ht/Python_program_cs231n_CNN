# -*- coding: UTF-8 -*-
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'  # 图像插值方式为最邻近插值
plt.rcParams['image.cmap'] = 'gray'  # 图像色彩空间为灰度(黑白)

# Load the raw CIFAR-10 data.
cifar10_dir = ("cs231n/datasets/cifar-10-batches-py")
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print ("Training data shape:", X_train.shape)
print ("Training labels shape: ", y_train.shape)
print ("Test data shape: ', X_test.shape")
print ('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    # y_train == y返回y_train矩阵中每个元素是否=y的布尔数组
    # idxs = 数组中为True的元素的索引
    idxs = np.flatnonzero(y_train == y)
    # 从idxs中选取samples_per_class个数的图片，组成矩阵赋值给idxs，replace=False表示不能有重复
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
# Subsample the data for more efficient code execution in this exercise
# 取前5000个训练样本
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# 取前500个测试样本
num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
# 将数据reshape，X_train.shape[0]表示行数，参数-1表示列数由程序自动推断
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print (X_train.shape, X_test.shape)

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

from time import time
# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print (dists.shape)





