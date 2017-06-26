import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class g(nn.Module):
    def __init__(self):
        super(g, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear1 = nn.Linear(1, 1, bias=False)
    def forward(self, input):
        return self.linear1(torch.exp(self.linear(x))+self.linear(x) -1)

x = np.linspace(0,1,100)
y = 2*(np.exp(3*x)+3*x-1)

x = np.reshape(x, (100, 1))
x = Variable(torch.FloatTensor(x),requires_grad = False)
y = Variable(torch.FloatTensor(y), requires_grad=False)

gg = g()

citi = nn.MSELoss()
optim = torch.optim.RMSprop(gg.parameters(), lr = 0.001)
for i in range(20000):
    optim.zero_grad()
    out = gg(x)
    loss = citi(out, y)
    print(loss.data.numpy())
    loss.backward()
    optim.step()
for para in gg.parameters():
    print(para.data)
