import torch
from torch.autograd import Variable

N,D_in,H,D_out =64,1000,100,10

x=Variable(torch.FloatTensor(torch.randn(N,D_in)),requires_grad=False)
y=Variable(torch.FloatTensor(torch.randn(N,D_out)),requires_grad=False)


model=torch.nn.Sequential(torch.nn.Linear(D_in,H),torch.nn.ReLU(),torch.nn.Linear(H,D_out))

loss_fn=torch.nn.MSELoss(size_average=False)
learning_rate=1e-4
optimier=torch.optim.Adam(model.parameters(),lr=learning_rate)

for i in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(i, loss)
    optimier.zero_grad()
    loss.backward()
    optimier.step()

print(loss)
