import torch
import torch.nn as nn
from torch.autograd import Variable
from readfile_ckim import FileRead
import numpy as np
import time
from convlstm import ConvLSTMCell
cuda = False
N = 1


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
N = 1
h1 = c1 = Variable(torch.zeros(N,10,3,49,49))
h2 = c2 = Variable(torch.zeros(N,20,2,22,22))
h3 = c3 = Variable(torch.zeros(N,40,1,8,8))



#三层网络初始化
lstm1 = ConvLSTMCell(input_num=1, hidden_num=10, output_num=1,batch_size=N,
                     kernal_shape_in=(2, 5, 5), kernal_shape_hidden=(1, 1, 1), stride_in=(1, 2, 2))
lstm2 = ConvLSTMCell(input_num=10, hidden_num=20, output_num=1,batch_size=N,
                     kernal_shape_in=(2, 7, 7), kernal_shape_hidden=(1, 1, 1),
                     kernal_shape_out=(2, 22, 22), stride_in=(1, 2, 2))
lstm3 = ConvLSTMCell(input_num=20, hidden_num=40, output_num=1,batch_size=N,
                     kernal_shape_in=(2, 8, 8), kernal_shape_hidden=(1, 1, 1),
                     kernal_shape_out=(1,8, 8), stride_in=(1, 2, 2))

lstm1.load_state_dict(torch.load('lstm1_epoch19')())
lstm2.load_state_dict(torch.load('lstm2_epoch19')())
lstm3.load_state_dict(torch.load('lstm3_epoch19')())

if cuda :
    lstm1.cuda()
    lstm2.cuda()
    lstm3.cuda()
    input, label = input.cuda(), label.cuda()
    h1, c1, h2, c2, h3, c3 = h1.cuda(), c1.cuda(), h2.cuda(), c2.cuda(), h3.cuda(), c3.cuda()
#测试
import csv
file = open('prediction.csv', 'a')
writer = csv.writer(file)
data = []
for t in range(size):
    for i in range(input.size()[1]):
        h1, c1 = lstm1(input[t*N:t*N+1,i,:,:,:,:], h1, c1)
        h2, c2 = lstm2(h1, h2, c2)
        h3, c3 = lstm3(h2, h3, h3)
    prediction = torch.squeeze(lstm3.get_output())
    criterion = torch.nn.MSELoss()
    loss = criterion(prediction, label[t])
    print('test loss :this is the loss of %d N,'%t,loss.data[0])
    for i in  prediction.data:
        writer.writerow([t,i])
file.close()
