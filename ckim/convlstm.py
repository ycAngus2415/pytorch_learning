import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
#卷积LSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, batch_size, kernal_shape_in=(2, 5, 5),
                 kernal_shape_hidden=(1, 1, 1), kernal_shape_out=(3,49,49), stride_in=(1, 2, 2), stride_out=(1,1,1)):
        super(ConvLSTMCell, self).__init__()
        self.batch_size = batch_size
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
        x = np.random.uniform(- 1.0/np.sqrt(self.batch_size), 1.0/np.sqrt(self.batch_size), shape)
        return torch.FloatTensor(x)

    def init(self):
        self
        for parameter in self.parameters():
            print(type(parameter.data), parameter.size())
            parameter.data = self.weight_init(parameter.size())
            parameter.reqires_grad = True


ff = ConvLSTMCell(10,20,20,40)

