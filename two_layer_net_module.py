import torch
from torch.autograd import Variable

N,D_in, H, D_out=64,1000,100,10

x=Variable(torch.FloatTensor(torch.randn(N,D_in)),requires_grad=False)
y=Variable(torch.FloatTensor(torch.randn(N,D_out)),requires_grad=False)

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)

    def forward(self,x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred

model=TwoLayerNet(D_in,H,D_out)
model.load_state_dict(torch.load('./model_epoch0'))
print(torch.load('./model_epoch0').keys())
loss_fn=torch.nn.MSELoss(size_average=False)

learning_rate=1e-4
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for i in range(500):
    y_pred=model(x)

    loss=loss_fn(y_pred,y)
    #print(i,loss)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
torch.save(model.state_dict(),'./model_epoch{0}'.format(0))
