import torch
import torch.nn as nn
from torch.autograd import Variable
import distribute as db
import argparse
import torchvision.utils
import torchvision.datasets as dset
import torch.utils.data as tud
import torchvision.transforms as transforms
import torch.nn.init

#设置参数
batchsize = 64
parse = argparse.ArgumentParser()
parse.add_argument('--datapath', required=True)
opt = parse.parse_args()
#数据初始化
input = torch.FloatTensor(batchsize, 3, 64, 64)
noise = torch.FloatTensor(batchsize, 100, 1, 1)
fixed_noise = torch.FloatTensor(batchsize, 100, 1, 1).normal_(0,1)
dataset = dset.CIFAR10(root=opt.datapath, download=True,
                       transform=transforms.Compose([
                           transforms.Scale(64),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                       ])
                       )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                         shuffle=True,
                                         num_workers=int(2)
                                         )
#标签
one = torch.FloatTensor([1])
mone = one * -1
#写一个权重初始化函数
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#生成器初始化
net_g=db.G_Net()
net_g.apply(weight_init)
#nn.init.normal(net_g.parameters())#parameters 不是一个tensor是一个迭代器,或者生成器，不能直接使用init
optimizer_g=torch.optim.RMSprop(net_g.parameters(), lr=1e-4)
#判别器初始化
net_d=db.D_Net()
#nn.init.normal(net_d.parameters())
net_d.apply(weight_init)
optimizer_d = torch.optim.RMSprop(net_d.parameters(), lr=1e-4)
#循环开始
for epoch in range(25):
    #数据载入
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            for p in net_d.parameters():
                p.requires_grad = True#将每个参数都变成需要导数的，
                p.data.clamp_(-1,1)
            data = data_iter.next()#这数据是一个list,0是图像数据，64*3*64*64,1是一个longtensor，还不清楚是啥，但是有用的部分时
            #print('this is data', data, '\n')
            i += 1
            #真实数据forward和backward
            input.resize_as_(data[0]).copy_(data[0])
            net_d.zero_grad()#在每次循环都要把梯度重置维0
            #print('this is input\n')
            #print(input,'\n')
            inputv = Variable(input)
            err_real = net_d(inputv)
            #print('this is err',err_real,'\n')
            err_real.backward(one)
            #伪数据
            noise.resize_(batchsize, 100, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = net_g(noisev)
            err_fake = net_d(fake)
            err_fake.backward(mone)
            #这阵求的损失函数是这个,这是wgan的精华
            err=err_real-err_fake
            optimizer_d.step()
            #训练G
            #这时候先要让d的梯度变为不可求梯度，不然
            for p in net_d.parameters():
                p.requires_grad = False
            net_g.zero_grad()#一定要重置为0
            noise.resize_(batchsize, 100, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = net_g(noisev)
            errg = net_d(fake)
            errg.backward(one)
            optimizer_g.step()
            #print('[%d/%d] 5dloss_d:%f loss_g:%f loss_d_real:%f loss_d_fake:%f'%(epoch, i, err.data, errg.data, err_real.data[0], err_fake.data[0]))
            print(r'[%d/%d][%d/%d]  '%(epoch, 25, i, len(dataloader)))
            print('\n', 'err',err)
            if i % 500 == 0:
                torchvision.utils.save_image(data[0], '{0}/real_samples.png'.format(opt.datapath))
                fake = net_g(Variable(fixed_noise))
                torchvision.utils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.datapath, epoch))

        torch.save(net_g.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.datapath,epoch))
        torch.save(net_d, '{0}/netD_epoch_{1}.pth'.format(opt.datapath, epoch))
