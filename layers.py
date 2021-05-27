import numpy as np

def xavier_init(in_units, out_units):
    return np.random.normal(loc=0.0, scale=np.sqrt(2/(in_units+out_units)), size=(in_units,out_units))


class ReLU():
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(0,input)
    
    def backward(self, input, grad_output):
        return grad_output*(input>=0)


class Dense():
    def __init__(self, in_units, out_units, lr=0.01):
        self.lr = lr
        self.w = xavier_init(in_units, out_units)
        self.b = np.zeros(out_units)
        
    def forward(self, input):
        return np.matmul(input,self.w) + self.b
    
    def backward(self, input, grad_out):
        grad_in = np.matmul(grad_out, self.w.T)
        
        grad_w = np.matmul(input.T, grad_out)
        self.w -= self.lr * grad_w
        
        grad_b = grad_out.mean(axis=0)*input.shape[0]
        self.b -= self.lr * grad_b
        
        return grad_in
