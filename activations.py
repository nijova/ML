import numpy as np
import matplotlib.pyplot as plt

def relu(x):
  return max(0,x)

xs = np.arange(-1,1,0.01)
relu_graph = [relu(x) for x in xs]

_ = plt.figure()
plt.vlines(0, 0, 1, colors='y')
plt.title('relu')
plt.plot(xs,relu_graph)
plt.show()

def leaky_relu(x, alpha=0.01):
  return max(alpha*x,x)

xs = np.arange(-10,10,0.01)
leaky_relu_graph = [leaky_relu(x, 0.05) for x in xs]

_ = plt.figure()
plt.vlines(0, -1, 10, colors='y')
plt.hlines(0, -10, 10, 'y')
plt.title('leaky relu, alpha=%1.2f' % alpha)
plt.plot(xs,leaky_relu_graph)
plt.show()


def elu(x, alpha):
  if x >= 0:
    return x
  else:
    return alpha * (np.e**x -1)

alpha = 2.
xs = np.arange(-10,10,0.01)
elu_graph = [elu(x, alpha) for x in xs]

_ = plt.figure()
plt.vlines(0, -2, 10, colors='y')
plt.hlines(0, -10, 10, colors='y')
plt.title('elu, alpha=%1.2f' % alpha) 
plt.plot(xs,elu_graph)
plt.show()


def sigmoid(x):
  return 1 / (1 + np.e**(-x))

xs = np.arange(-10,10,0.01)
sigmoid_graph = [sigmoid(x) for x in xs]

_ = plt.figure()
plt.vlines(0, 0, 1, colors='y')
plt.hlines(0, -10, 10, colors='y')
plt.title('sigmoid')
plt.plot(xs,sigmoid_graph)
plt.show()

def tanh(x):
  return np.tanh(x)

xs = np.arange(-10,10,0.01)
tanh_graph = [tanh(x) for x in xs]

_ = plt.figure()
plt.vlines(0, -1, 1, colors='y')
plt.hlines(0, -10, 10, colors='y')
plt.title('tanh')
plt.plot(xs,tanh_graph)
plt.show()

def softmax(logits):
  logits = np.array(logits)
  return np.e ** logits / np.sum(np.e ** logits)

#softmax([-1,0,1])

def logsoftmax(logits):
  return np.log(softmax(logits))

logsoftmax([1,3,2])
