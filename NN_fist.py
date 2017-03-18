import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)#一个输入，六个输出，5*5的方针
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)#an affine operation
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))#使用2*2的卷积核过滤
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,self.num_flat_features(x))

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features

# net=Net()
# print(net)
#
# params=list(net.parameters())
# print(len(params))
# print(params[0].size())
#
# input=Variable(torch.rand(1,1,32,32))
# out=net(input)
# print(out)
#
# net.zero_grad()
#
#
#
# target=Variable(torch.randn(1,10))
# criterion=nn.MSELoss()
# loss=criterion(out,target)
# print(loss)
# print(loss.creator)
# print(net.conv1.bias.grad)




#
# optimizer=optim.SGD(net.parameters(),lr=0.01)
#
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# #test
# out1=net(input)
# loss1=criterion(out1,target)
# print(loss1)

transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
                                        shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=4,
                                        shuffle=False,num_workers=2)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def imshow(img):
    img=img/2+0.5
    nping=img.numpy()
    plt.imshow(np.transpose(nping,(1,2,0)))


# dataiter=iter(trainloader)
# images,labels=dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# plt.show()
# print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

net=Net()

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


for epoch in range(2):

    running_loss=0
    for i, data in enumerate(trainloader,0):
        inputs,labels=data

        inputs,labels=Variable(inputs),Variable(labels)

        optimizer.zero_grad()

        output=net(inputs)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.data[0]
        if i%2000==1999:
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print('finished traning')


correct =0
total = 0
for data in testloader:
    images,labels=data
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs.data,1)
    total+=labels.size(0)
    correct +=(predicted==labels).sum()

print('accuracy:%d %%'%(100*correct/total))
