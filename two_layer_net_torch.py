import torch

N,D_in,H,D_out=64,1000,100,10
x=torch.FloatTensor(torch.randn(N,D_in))
y=torch.FloatTensor(torch.randn(N,D_out))

w1=torch.FloatTensor(torch.randn(D_in,H))
w2=torch.FloatTensor(torch.randn(H,D_out))


learning_rate=1e-6#知道学习率多重要了吧
for i in range(450):
    h=x.mm(w1)
    h_relu=h.clamp(min=0)
    y_out=h_relu.mm(w2)

    loss=(y_out-y).pow(2).sum()
    print(i,loss)
    grad_y=2*(y_out-y)
    grad_w2= h_relu.t().mm(grad_y)
    grad_h_relu=grad_y.mm(w2.t())
    grad_h_relu[h<0]=0
    grad_w1=x.t().mm(grad_h_relu)

    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

print(loss)
