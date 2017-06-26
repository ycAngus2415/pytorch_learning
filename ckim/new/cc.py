import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

class BiJin2(nn.Module):
    def __init__(self):
        super(BiJin2, self).__init__()
        self.linear1 = nn.Linear(4, 60)
        self.linear2 = nn.Linear(60, 200)
        self.linear3 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(self.linear3(self.sigmoid(self.linear2(self.sigmoid(self.linear1(input))))))


x = np.linspace(0, 10, 100)
y = np.sin(x)
x_tensor = torch.FloatTensor(x)
x_retensor = x_tensor.unfold(0, 5, 2)
y_tensor = torch.sin(x_retensor)
y_retensor = (y_tensor+1)/2
plt.figure()
plt.subplot(2,2, 1)
plt.plot(x, (y+1)/2, 'o')
x_v = Variable(y_retensor[:, :4], requires_grad=False)
y_v = Variable(y_retensor[:, -1], requires_grad=False)
x_re = y_retensor[0,:3]
y_re = y_retensor[:,3:4]

bj = BiJin2()
cri = torch.nn.MSELoss()
optim = torch.optim.RMSprop(bj.parameters(), lr=0.0003)
loss_d = []
for i in range(200):
    out = bj(x_v)
    loss = cri(out, y_v)
    loss_d.append(loss.data[0])
    loss.backward()
    optim.step()
out = bj(x_v)
y_out = np.concatenate((y_re.numpy(), out.data.numpy()), 1)
y_out_re = np.reshape(y_out, 96)
y_out_end = np.concatenate((x_re.numpy(), y_out_re), 0)
plt.plot(x[:-1], y_out_end)

plt.subplot(2,2,2)
xxx = np.linspace(-10, 0, 100)
y = np.sin(xxx)
plt.plot(xxx, (y+1)/2, 'o')
x = torch.FloatTensor(xxx)
x = x.unfold(0, 5, 2)
y = torch.sin(x)
y_retensor = (y+1)/2
xx = Variable(y_retensor[:, :4], requires_grad=False)
yy = Variable(y_retensor[:, -1], requires_grad=False)
x_re = y_retensor[0,:3]
y_re = y_retensor[:,3:4]
out = bj(xx)
y_out = np.concatenate((y_re.numpy(), out.data.numpy()), 1)
y_out_re = np.reshape(y_out, 96)
y_out_end = np.concatenate((x_re.numpy(), y_out_re), 0)
plt.plot(xxx[:-1], y_out_end)
plt.subplot(2,2,3)
plt.plot(range(200), loss_d, 'o')
plt.savefig('./figure.png')
plt.show()
