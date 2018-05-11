import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  #初始化梯度值
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  #计算损失和梯度
  num_classes = W.shape[1]#C
  num_train = X.shape[0]  #N
  loss = 0.0
  for i in range(num_train):
    #C个类别全打分
    scores = X[i].dot(W)
    #取出正确分类的分数
    correct_class_score = scores[y[i]]
    #类别数循环
    for j in range(num_classes):
      #跳过正确的分类，计算损失loss
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i]
        dW[:,j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  # 损失、未加正则项梯度求平均
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  # 损失增加L2范数正则项，reg为正则项超参数系数,梯度增加正则项部分
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.
  # 计算损失函数的梯度存储在dW中                                                  #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.
  # 计算loss时同时计算导数会相对更容易，因此你可以在计算loss的时候修改一些代码           #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  # 运用向量化的结构计算SVM损失，结果存储在loss中                                   #
  #############################################################################
  num_classes = W.shape[1]#C
  num_train = X.shape[0]  #N
  #为数据打分环节
  scores = X.dot(W)
  #loss公式计算前部分
  margin = np.maximum(0, scores - scores[range(num_train), y].reshape(-1,1) + 1)
  # margin = (margin >0) * margin
  margin[range(num_train),y] = 0
  #计算梯度的前半部分
  caclu_amut = np.zeros((num_train,num_classes))
  caclu_amut[margin>0] = 1
  caclu_amut[range(num_train), y] = 0
  caclu_amut[range(num_train), y] = -np.sum(caclu_amut,axis=1)
  dW = np.dot(X.T,caclu_amut)

  loss += np.sum(margin)
  #求均值，增加正则项
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW/num_train + reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  # 运用向量化的结构计算SVM损失的梯度，结果存储在dW中                              #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.
  # 提示：不用再胡乱的计算梯度，利用计算损失时的一些中间值可能更容易                    #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
