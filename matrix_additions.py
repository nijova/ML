import numpy as np

# entrywise sum

a=np.arange(12).reshape(3,4)
b=np.arange(12).reshape(3,4)

def entrywise_sum(m,n):
  result = []
  for row in range(m.shape[0]):
    for col in range(m.shape[1]):
     result.append(m[row,col] + n[row,col])
  return np.array(result).reshape(m.shape)

print(entrywise_sum(a,b))
print(a + b)
