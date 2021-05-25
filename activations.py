import numpy as np
import matplotlib.pyplot as plt

def relu(x):
  return max(0,x)

xs = np.arange(-1,1,0.01)
relu_graph = [relu(x) for x in xs]

_ = plt.figure()
plt.plot(xs,relu_graph)
plt.show()
