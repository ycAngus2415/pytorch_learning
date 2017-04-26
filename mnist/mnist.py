import torch
from torch.autograd import Variable
import torch.nn as nn
import trochvision
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.utils
import torchvision.transforms as transforms

class Mnist_Net(nn.Module):
    def __init__(self):
        super(Mnist_Net, self).__init__()
        self.line = nn.Linear(784,10)
        self.sofmax = nn.Softmax()

    def forwart(self, input):
        self.out = self.sofmax(self.line(input))
        return out
dataset = datasets.MNIST('../data', train=True, download=True,
                         transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                         ]))
dataloader = torchvision.utils.dataloader(dataset, batch_size=64,
                                          shuffle=True)
testset = datasets.MNIST('../data', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
mnist = Mnist_Net()
criminater = nn.BCELoss()
optimer = optim.SGD(mnist.parameters(), lr=1e-4, momentum=0.5))
for epoch in range(2):
    for i, (data, target) in enumerate(dataloader):
        input, label = Variable(data), Variable(target)
        optimer.zero_grad()
        out = mnist(input)
        err = criminater(out, label)
        loss.backward()
        optimer.step()
torchvision.utils.save_image(
