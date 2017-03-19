import torch
from torch.autograd import Variable

#先定义nn。module
class DynamiNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(DynamiNet,self).__init__()
        self.input_linear1=torch.nn.Linear(D_in,H)
        self.h_linear=torch.nn.Linear(H,H)
        self.output_linear=torch.nn.Linear(H,D_out)

    def forward(self,x):
        h_relu=self.input_linear1(x).clamp(min=0)
        for _ in range(3):
            h_relu=self.h_linear(h_relu).clamp(min=0)
        y_pred=self.output_linear(h_relu)
        return y_pred

#定义数据
N,D_in, H, D_out=64,1000,100,10

x=Variable(torch.FloatTensor(torch.randn(N,D_in)),requires_grad=False)
y=Variable(torch.FloatTensor(torch.randn(N,D_out)),requires_grad=False)

model=DynamiNet(D_in,H,D_out)

loss_fn=torch.nn.MSELoss(size_average=False)

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

for i in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(i,loss)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
