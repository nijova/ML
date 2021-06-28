import numpy as np

def kronecker_product(m,n):
  result = []
  for row in range(m.shape[0]):
    for r in range(n.shape[0]):
      for col in range(m.shape[1]):
        for c in range(n.shape[1]):
          result.append(m[row,col] * n[r,c])
  return np.array(result).reshape(m.shape[0]*n.shape[0], m.shape[1]*n.shape[1])

a = np.array([1,2,3,4]).reshape(2,2)
b = np.array([0,5,6,7]).reshape(2,2)

print(kronecker_product(a,b))
