import numpy as np

def matmul(m,n):
  result = []
  for r in range(m.shape[0]):
    for c in range(n.shape[1]):
      sum = 0
      for x in range(len(m[r])):
        sum += m[r,x] * n[x,c]
      result.append(sum)
  return np.array(result).reshape(m.shape[0], n.shape[1])

a=np.arange(16).reshape(8,2)
b=np.arange(8).reshape(2,4)

print(matmul(a, b))

print(np.matmul(a, b))
