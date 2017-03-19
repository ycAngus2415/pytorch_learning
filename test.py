import torch
from torch.autograd import Variable

t=Variable(torch.FloatTensor(torch.rand(3,4)),requires_grad=True)
k=Variable(torch.FloatTensor(torch.rand(4)),requires_grad=False)
r=t.mv(k)
print(r)
y=r.sum()
print(y)

y.backward()
print(t.grad.data)
print(t.grad.data)
