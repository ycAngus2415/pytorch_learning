import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from readfile_ckim import FileRead
#数据的获取，在自己写的readfile_ckim.py 中
ff = FileRead()
ff.readfile()
x =[]
y = []
o=0
for train in ff.train_data.values():
    x.append(train['data'])
    y.append(float(train['label']))
    o += 1
x = np.array(x)
x = np.reshape(x, (o,15,1,4,101,101))
print(np.shape(x))
y = np.array(y)
input = Variable(torch.FloatTensor(x), requires_grad=False)
label = Variable(torch.FloatTensor(y), requires_grad=False)
#记录总的数据size ， N表示batch size
size = o
if size > 100:
    N = 100
else:
    N = size

#卷积LSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, kernal_shape_in=(2, 5, 5),
                 kernal_shape_hidden=(1, 1, 1), kernal_shape_out=(3,49,49), stride_in=(1, 2, 2), stride_out=(1,1,1)):
        super(ConvLSTMCell, self).__init__()
        self.conv3_W_xi = nn.Conv3d(input_num, hidden_num, kernal_shape_in, stride_in)
        self.conv3_W_hi = nn.Conv3d(hidden_num, hidden_num, kernal_shape_hidden)
        self.sigmoid = nn.Sigmoid()
        self.conv3_W_xf = nn.Conv3d(input_num, hidden_num, kernal_shape_in, stride_in)
        self.conv3_W_hf = nn.Conv3d(hidden_num, hidden_num, kernal_shape_hidden)
        self.conv3_W_xo = nn.Conv3d(input_num, hidden_num, kernal_shape_in, stride_in)
        self.conv3_W_xc = nn.Conv3d(input_num, hidden_num, kernal_shape_in, stride_in)
        self.conv3_W_ho = nn.Conv3d(hidden_num, hidden_num, kernal_shape_hidden)
        self.conv3_W_hc = nn.Conv3d(hidden_num, hidden_num, kernal_shape_hidden)
        self.tanh = nn.Tanh()
        self.conv3_w_out = nn.Conv3d(hidden_num, output_num, kernal_shape_out, stride_out)
        self.init()

    def forward(self, x, h, c):
        i = self.sigmoid(self.conv3_W_xi(x) + self.conv3_W_hi(h))
        f = self.sigmoid(self.conv3_W_xf(x) + self.conv3_W_hf(h))
        o = self.sigmoid(self.conv3_W_xo(x) + self.conv3_W_ho(h))
        g = self.sigmoid(self.conv3_W_xc(x) + self.conv3_W_hc(h))
        self.c = c = torch.mul(f, c) + torch.mul(i, self.tanh(g))
        self.h = h = torch.mul(o, self.tanh(c))
        return h, c
    def get_output(self):
        return self.conv3_w_out(self.h)

    def weight_init(self, shape):
        x = np.random.uniform(- 1.0/np.sqrt(N), 1.0/np.sqrt(N), shape)
        return torch.FloatTensor(x)

    def init(self):
        for parameter in self.parameters():
            parameter.data = self.weight_init(parameter.size())
            parameter.reqires_grad = True


#三层网络初始化
lstm1 = ConvLSTMCell(input_num=1, hidden_num=10, output_num=1,
                     kernal_shape_in=(2, 5, 5), kernal_shape_hidden=(1, 1, 1), stride_in=(1, 2, 2))
lstm2 = ConvLSTMCell(input_num=10, hidden_num=20, output_num=1,
                     kernal_shape_in=(2, 7, 7), kernal_shape_hidden=(1, 1, 1),
                     kernal_shape_out=(2, 22, 22), stride_in=(1, 2, 2))
lstm3 = ConvLSTMCell(input_num=20, hidden_num=40, output_num=1,
                     kernal_shape_in=(2, 8, 8), kernal_shape_hidden=(1, 1, 1),
                     kernal_shape_out=(1,8, 8), stride_in=(1, 2, 2))
h1 = c1 = Variable(torch.zeros(N,10,3,49,49))
h2 = c2 = Variable(torch.zeros(N,20,2,22,22))
h3 = c3 = Variable(torch.zeros(N,40,1,8,8))


#训练
for epoch in range(20):

    for t in range(N/size):
        for i in range(input.size()[1]):
            h1, c1 = lstm1(input[:,i,t*N:t*N+1,:,:,:], h1, c1)
            h2, c2 = lstm2(h1, h2, c2)
            h3, c3 = lstm3(h2, h3, h3)
        prediction = lstm3.get_output()
        optimizer = torch.optim.SGD(lstm1.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(prediction), label)
        loss.backward(retain_variables=True)
        optimizer.step()
    print('this is the loss of %d epoch'%epoch,loss)
    torch.save(lstm1.state_dict,'./epoch{0}'.format(epoch))

#x =[]
#x.append(ff.train_data['train3']['data'])
#x.append(ff.train_data['train4']['data'])
#x = np.array(x)
#x = np.reshape(x, (2,15,1,4,101,101))
#print(np.shape(x))
#y = []
#y.append(float(ff.train_data['train3']['label']))
#y.append(float(ff.train_data['train4']['label']))
#y = np.array(y)
#input = Variable(torch.FloatTensor(x), requires_grad=False)
#label = Variable(torch.FloatTensor(y), requires_grad=False)
#for i in range(input.size()[1]):
#    h1, c1 = lstm1(input[:,i,:,:,:,:], h1, c1)
#    h2, c2 = lstm2(h1, h2, c2)
#    h3, c3 = lstm3(h2, h3, h3)
#prediction = lstm3.get_output()
#loss = criterion(torch.squeeze(prediction), label)
#print(loss)
