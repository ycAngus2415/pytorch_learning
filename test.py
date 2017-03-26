import torch
from torch.autograd import Variable
import random
import numpy as np

x=Variable(torch.FloatTensor(torch.randn(1,10)),requires_grad=False)
w=Variable(torch.FloatTensor(torch.randn(10,100)),requires_grad=True)
y=x.mm(w)
print(y)
y=torch.transpose(y,0,1)
print(y)
print(x.data)
y.backward(torch.FloatTensor([1]))
print(w.grad)
