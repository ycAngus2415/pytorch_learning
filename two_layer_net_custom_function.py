import torch
from torch.autograd import Variable

dtype=torch.FloatTensor
class ReLU(torch.autograd.Function):

    def forward(self,input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self,grad_output):
        input, =self.saved_tensors
        grad_input=grad_output.clone()
        grad_input[input<0]=0
        return grad_input

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(D_in,N).type(dtype),requires_grad=False)
y=Variable(torch.randn(D_out,N).type(dtype),requires_grad=False)

w1=Variable(torch.randn(H,D_in).type(dtype),requires_grad=True)
w2=Variable(torch.randn(D_out,H).type(dtype),requires_grad=True)

learning=1e-6
for i in range(500):
    relu=ReLU()
    y_pred=w2.mm(relu(w1.mm(x)))
    loss=(y_pred-y).pow(2).sum()


    print(i,loss.data[0])

    loss.backward()
    w1.data -=learning*w1.grad.data
    w2.data -= learning*w2.grad.data

print(loss)
