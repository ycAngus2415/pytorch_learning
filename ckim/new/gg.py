import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

class BiJin2(nn.Module):
    def __init__(self):
        super(BiJin2, self).__init__()
        self.linear1 = nn.Linear(5, 60)
        self.linear2 = nn.Linear(60, 200)
        self.linear3 = nn.Linear(200, 2)
        self.sigmoid = nn.Sigmoid()
        nn.Conv3d

    def forward(self, input):
        return self.sigmoid(self.linear3(self.sigmoid(self.linear2(self.sigmoid(self.linear1(input))))))

x = np.linspace(0, 10, 100)
x1 = np.linspace(5, 15, 100)
y = np.sin(x)
y1 = np.cos(x1)
x_tensor = torch.FloatTensor(x)
x1_tensor = torch.FloatTensor(x1)
x_retensor = x_tensor.unfold(0, 7, 2)
x1_retensor = x1_tensor.unfold(0, 7, 2)
y_tensor = torch.sin(x_retensor)
y1_tensor = torch.cos(x_retensor)
y_retensor = (y_tensor+1)/2
y1_retensor = (y1_tensor +1)/2
plt.figure()
plt.subplot(2,2, 1)
plt.plot(x, (y+1)/2, 'o')
plt.plot(x1, (y1+1)/2, 'o')
x_v = y_retensor[:, :5]
y_v = y_retensor[:, 5:7]
xx_v = x_retensor[:, 5:7]
x1_v = y1_retensor[:,:5]
y1_v = y1_retensor[:, 5:7]
xx1_v = x1_retensor[:, 5:7]
x_cat = torch.cat((x_v, x1_v), 0)
y_cat = torch.cat((y_v, y1_v), 0)
xx_cat = torch.cat((xx_v, xx1_v), 0)
perm = torch.randperm(94)
x_v = Variable(x_cat[perm])
y_v = Variable(y_cat[perm])
xx_v = xx_cat[perm]

bj = BiJin2()
cri = torch.nn.MSELoss()
optim = torch.optim.RMSprop(bj.parameters(), lr=0.0003)
loss_d = []
for i in range(1000):
    out = bj(x_v)
    loss = cri(out, y_v)
    loss_d.append(loss.data[0])
    loss.backward()
    optim.step()
out = bj(x_v)

plt.plot(xx_v.numpy()[:,0], out.data.numpy()[:,0], 'o')
plt.plot(xx_v.numpy()[:,1], out.data.numpy()[:,1], 'o')

x = np.linspace(-5, 15, 100)
x1 = np.linspace(10, 20, 100)
y = np.sin(x)
y1 = np.cos(x1)
x_tensor = torch.FloatTensor(x)
x1_tensor = torch.FloatTensor(x1)
x_retensor = x_tensor.unfold(0, 7, 2)
x1_retensor = x1_tensor.unfold(0, 7, 2)
y_tensor = torch.sin(x_retensor)
y1_tensor = torch.cos(x1_retensor)
y_retensor = (y_tensor+1)/2
y1_retensor = (y1_tensor +1)/2
plt.subplot(2,2, 2)
plt.plot(x, (y+1)/2, 'o')
plt.plot(x1, (y1+1)/2, 'o')
x_v = y_retensor[:, :5]
y_v = y_retensor[:, 5:7]
xx_v = x_retensor[:, 5:7]
x1_v = y1_retensor[:,:5]
y1_v = y1_retensor[:, 5:7]
xx1_v = x1_retensor[:, 5:7]
x_cat = torch.cat((x_v, x1_v), 0)
y_cat = torch.cat((y_v, y1_v), 0)
xx_cat = torch.cat((xx_v, xx1_v), 0)
perm = torch.randperm(94)
x_v = Variable(x_cat[perm])
y_v = Variable(y_cat[perm])
xx_v = xx_cat[perm]

out = bj(x_v)
plt.plot(xx_v.numpy()[:,0], out.data.numpy()[:,0], 'o')
plt.plot(xx_v.numpy()[:,1], out.data.numpy()[:,1], 'o')
loss = cri(out, y_v)
print(loss.data[0])
plt.subplot(2,2,3)
plt.plot(range(1000), loss_d, 'o')
plt.savefig('./figure_4.png')
plt.show()
