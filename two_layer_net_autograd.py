import torch
from torch.autograd import Variable

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.FloatTensor(torch.randn(N,D_in)),requires_grad=False)
y=Variable(torch.FloatTensor(torch.randn(N,D_out)),requires_grad=False)

w1=Variable(torch.FloatTensor(torch.randn(D_in,H)),requires_grad=True)
w2=Variable(torch.FloatTensor(torch.randn(H,D_out)),requires_grad=True)

learning_rate=3*1e-7#there is some problem
for t in range(500):
    y_pred=x.mm(w1).clamp(min=0).mm(w2)
    loss=(y_pred-y).pow(2).sum()
    print(t,loss.data[0])
    loss.backward()
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data

print(loss.data)
print(type(w1.grad))
