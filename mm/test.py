import torch
import scipy
from torch.autograd import Variable

one = torch.FloatTensor([2])
x = torch.FloatTensor(30)
x = x.normal_(0, 1)
x = Variable(x, requires_grad=True)
y = 2*x
y.backward(one)
print(x.size())
print(y.size())
print(one)
print(x.grad.data)
