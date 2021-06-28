import numpy as np


########## entrywise sum

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


########## direct sum

a=np.arange(9).reshape(3,3)
b=np.arange(20).reshape(5,4)

def direct_sum(m,n):
  new_shape = (m.shape[0] + n.shape[0], m.shape[1] + n.shape[1])
  result = np.zeros(new_shape)
  for row in range(m.shape[0]):
    for col in range(m.shape[1]):
      result[row,col] = m[row,col]
  for row in range(n.shape[0]):
    for col in range(n.shape[1]):
      result[m.shape[0]+row ,m.shape[1]+col] = n[row,col]
  return result

print(direct_sum(a,b))


