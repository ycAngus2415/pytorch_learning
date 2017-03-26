import torch
from torch.autograd import Variable
import torch.nn as nn

class G_Net(nn.Module):
    '''这是生成网络
    '''
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv=nn.ConvTranspose2d(100,512,4,1,0)
        self.conv1 = nn.ConvTranspose2d(512,256,4,2,1)
        self.conv2 = nn.ConvTranspose2d(256,128,4,2,1)
        self.conv3 = nn.ConvTranspose2d(128,64,4,2,1)
        self.conv4 = nn.ConvTranspose2d(64,3,4,2,1)
        self.relu = nn.ReLU(True)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.batchnorm = nn.BatchNorm2d(512)
        self.batchnomr1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.tanh=nn.Tanh()
    def forward(self, x):
        out = self.relu(self.batchnorm(self.conv(x)))
        out = self.relu1(self.batchnomr1(self.conv1(out)))
        out = self.relu2(self.batchnorm2(self.conv2(out)))
        out = self.relu3(self.batchnorm3(self.conv3(out)))
        out = self.tanh(self.conv4(out))
        return out


class D_Net(nn.Module):
    '''这是对抗网络
    '''
    def __init__(self):
        super(D_Net,self).__init__()
        self.conv =nn.Conv2d(3,64,4,2,1)
        self.conv1 = nn.Conv2d(64,128,4,2,1)
        self.conv2 = nn.Conv2d(128,256,4,2,1)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv4 = nn.Conv2d(512, 1, 4, 1, 0)
        self.lrelu = nn.LeakyReLU(0.2,inplace=True)
        self.lrelu1 = nn.LeakyReLU(0.2,inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.2,inplace=True)
        self.lrelu3 = nn.LeakyReLU(0.2,inplace=True)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(128)

    def forward(self,x):
        out = self.lrelu(self.conv(x))
        out = self.lrelu1(self.batchnorm2(self.conv1(out)))
        out = self.lrelu2(self.batchnorm3(self.conv2(out)))
        out = self.lrelu3(self.batchnorm4(self.conv3(out)))
        out = self.conv4(out)
        return out.mean(0)
