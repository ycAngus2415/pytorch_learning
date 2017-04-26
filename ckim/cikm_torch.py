from matl_ckim import FileRead
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
rnn = torch.nn.LSTMCell(20, 20)
x = Variable(torch.rand(5,3,20))
y = torch.sin(x)
h_0 = Variable(torch.zeros(3,20))
c_0 = Variable(torch.zeros(3,20))
output = []
critization = torch.nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.1)
loss_total = 0
for j in range(200):
    h_0 = Variable(torch.zeros(3,20))
    c_0 = Variable(torch.zeros(3,20))
    loss_total = 0
    optimizer.zero_grad()
    for i in range(5):
        h_0, c_0 = rnn(x[i], (h_0, c_0))
        output.append(h_0)
        loss = critization(h_0, y[i])
        loss_total += loss
    print(loss_total.data[0])

    loss_total.backward()
    optimizer.step()


