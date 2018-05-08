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
num_test = 100
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


# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print (dists.shape)

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
#统计预测正确样本标签的个数
num_correct = np.sum(y_test_pred == y_test)
#计算KNN预测精度
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
#使用局部向量化的一层循环提高计算距离量的速度
dists_one = classifier.compute_distances_one_loop(X_test)
print(dists_one.shape)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
#使用Frobenius 规范验证两种求解距离结果的相似性，
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# Let's compare how fast the implementations are
def time_function(f, *args):

    # Call a function f with args and return the time (in seconds) that it took to execute.
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# you should see significantly faster performance with the fully vectorized implementation

"""
We have implemented the k-Nearest Neighbor classifier but we set the value k = 5
arbitrarily. We will now determine the best value of this hyperparameterwith cro
ss-validation.
"""
# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
# 这里没有切片测试数据集，整体参与统计                                               #
################################################################################
classifier.train(X_train, y_train)
dists = classifier.compute_distances_no_loops(X_test)
for i in k_choices:
    y_test_pred = classifier.predict_labels(dists,i)
    #统计预测正确样本标签的个数
    num_correct = np.sum(y_test_pred == y_test)
    #计算KNN预测精度
    accuracy = float(num_correct) / num_test
    k_to_accuracies[i] = accuracy
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    print('k = %d, accuracy = %f' % (k, k_to_accuracies[k]))