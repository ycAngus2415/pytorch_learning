import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from readfile_ckim import FileRead
from convlstm import ConvLSTMCell
#数据的获取，在自己写的readfile_ckim.py 中
cuda = False
ff = FileRead()
ff.readfile('./data/data_sample.txt')
x =[]
y = []
o=0
for train in ff.train_data.values():
    x.append(train['data'])
    y.append(float(train['label']))
    o += 1
x = np.array(x)
x = np.reshape(x, (o,15,1,4,101,101))
y = np.array(y)
input = Variable(torch.FloatTensor(x), requires_grad=False)
label = Variable(torch.FloatTensor(y), requires_grad=False)
#记录总的数据size ， N表示batch size
size = o
if size > 100:
    N = 100
else:
    N = size


#三层网络初始化
lstm1 = ConvLSTMCell(input_num=1, hidden_num=10, output_num=1,batch_size=N,
                     kernal_shape_in=(2, 5, 5), kernal_shape_hidden=(1, 1, 1), stride_in=(1, 2, 2))
lstm2 = ConvLSTMCell(input_num=10, hidden_num=20, output_num=1,batch_size=N,
                     kernal_shape_in=(2, 7, 7), kernal_shape_hidden=(1, 1, 1),
                     kernal_shape_out=(2, 22, 22), stride_in=(1, 2, 2))
lstm3 = ConvLSTMCell(input_num=20, hidden_num=40, output_num=1,batch_size=N,
                     kernal_shape_in=(2, 8, 8), kernal_shape_hidden=(1, 1, 1),
                     kernal_shape_out=(1,8, 8), stride_in=(1, 2, 2))
h1 = c1 = Variable(torch.zeros(N-1,10,3,49,49))
h2 = c2 = Variable(torch.zeros(N-1,20,2,22,22))
h3 = c3 = Variable(torch.zeros(N-1,40,1,8,8))
#cuda
if cuda :
    lstm1.cuda()
    lstm2.cuda()
    lstm3.cuda()
    input, label = input.cuda(), label.cuda()
    h1, c1, h2, c2, h3, c3 = h1.cuda(), c1.cuda(), h2.cuda(), c2.cuda(), h3.cuda(), c3.cuda()

load = False
if load :
    lstm1.load_state_dict(torch.load('./lstm1_epoch8'))
    lstm2.load_state_dict(torch.load('./lstm2_epoch8'))
    lstm3.load_state_dict(torch.load('./lstm3_epoch8'))

optimizer = torch.optim.Adam(lstm1.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
#训练
for epoch in range(20):

    for t in range(size//N):
        optimizer.zero_grad()
        for i in range(input.size()[1]):
            h1, c1 = lstm1(input[t*N:(t+1)*N-1,i,:,:,:,:], h1, c1)
            h2, c2 = lstm2(h1, h2, c2)
            h3, c3 = lstm3(h2, h3, h3)
        prediction = lstm3.get_output()
        loss = criterion(torch.squeeze(prediction), label[t*N:(t+1)*N-1])
        loss.backward(retain_variables=True)
        optimizer.step()
    print('train loss : this is the loss of %d epoch,'%epoch,loss.data[0])
    torch.save(lstm1.state_dict(),'./lstm1_epoch{0}'.format(epoch))
    torch.save(lstm2.state_dict(),'./lstm2_epoch{0}'.format(epoch))
    torch.save(lstm3.state_dict(),'./lstm3_epoch{0}'.format(epoch))

#ff = FileRead()
#ff.readfile()
#x =[]
#y = []
#o=0
#for train in ff.train_data.values():
#    x.append(train['data'])
#    y.append(float(train['label']))
#    o += 1
#x = np.array(x)
#x = np.reshape(x, (o,15,1,4,101,101))
#y = np.array(y)
#input = Variable(torch.FloatTensor(x), requires_grad=False)
#label = Variable(torch.FloatTensor(y), requires_grad=False)
##记录总的数据size ， N表示batch size
#size = o
#if size > 100:
#    N = 100
#else:
#    N = size
#N =1
#h1 = c1 = Variable(torch.zeros(N,10,3,49,49))
#h2 = c2 = Variable(torch.zeros(N,20,2,22,22))
#h3 = c3 = Variable(torch.zeros(N,40,1,8,8))
#
#
#if cuda :
#    lstm1.cuda()
#    lstm2.cuda()
#    lstm3.cuda()
#    input, label = input.cuda(), label.cuda()
#    h1, c1, h2, c2, h3, c3 = h1.cuda(), c1.cuda(), h2.cuda(), c2.cuda(), h3.cuda(), c3.cuda()
##测试
#for t in range(size):
#    for i in range(input.size()[1]):
#        h1, c1 = lstm1(input[t*N:t*N+1,i,:,:,:,:], h1, c1)
#        h2, c2 = lstm2(h1, h2, c2)
#        h3, c3 = lstm3(h2, h3, h3)
#    prediction = lstm3.get_output()
#    criterion = torch.nn.MSELoss()
#    loss = criterion(torch.squeeze(prediction), label[t])
#    print('test loss :this is the loss of %d N,'%t,loss.data[0])
