import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision

class BiJin(nn.Module):
    def __init__(self):
        super(BiJin, self).__init__()
        self.linear1 = nn.Linear(1, 200)
        self.linear2 = nn.Linear(200, 1)
        self.linear3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        return self.sigmoid(self.linear2(self.sigmoid(self.linear1(input))))

xx = np.linspace(-np.pi, np.pi, 100)
xx = np.array(xx)
xx = np.reshape(xx, (100, 1))
yy = (np.sin(xx)+1)/2
#xx = (xx + np.pi)/(2*np.pi)
yy = np.reshape(yy, (100, 1))
x = Variable(torch.FloatTensor(xx), requires_grad=False)
y = Variable(torch.FloatTensor(yy), requires_grad=False)
cri = nn.MSELoss()
bj = BiJin()
opti = torch.optim.RMSprop(bj.parameters(), lr=0.0003)
loss_d = []
for i in range(5000):
    out = bj(x)
    loss = cri(out, y)
    loss_d.append(loss.data[0])
    loss.backward()
    opti.step()
out = bj(x)
ddd = out.data.numpy()
plt.figure()
plt.subplot(2,2,1)
plt.plot(xx, yy, 'o')
plt.plot(xx, ddd, 'o')
plt.subplot(2, 2, 2)
plt.plot(range(5000), loss_d, 'o')
x = np.linspace(-5* np.pi , 5* np.pi, 100)
y = (np.sin(x)+1)/2
x = np.reshape(x, (100, 1))

plt.subplot(2,2,3)
plt.plot(x, y, 'o')
x = Variable(torch.FloatTensor(x))
out = bj(x)
plt.plot(x.data.numpy(), out.data.numpy(), 'o')
plt.savefig('./figure_2.png')
plt.show()
