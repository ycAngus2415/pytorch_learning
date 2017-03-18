import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
x=torch.Tensor(5,3)
x=torch.rand(5,3)
print(x)

print(x.size())
y=torch.rand(5,3)
print(x+y)
z=torch.add(x,y)
print(z)
result=torch.Tensor(5,3)
d=torch.add(x,y,out=result)
print(d)
print('\n')
print(result)

print(torch.numel(x))
print(torch.eye(3))#和numpy 是一样的

a=np.random.randn(3,4)
print(a)
b=torch.from_numpy(a)
print(b)

a=torch.linspace(3,5,100)
print(a)

b=torch.logspace(3,5,100)
print(b)
# plt.plot(a.numpy(),b.numpy())
# plt.show()
a=torch.ones(3)
print(a)
b=torch.ones((3,2))
print(b)

a=torch.rand(3,4)
b=torch.randn(3,4)
print(a,'\n',b)
torch.cat

a=torch.randperm(5)
print(a)

a=torch.range(1,10,2)
print(a)

a=torch.zeros((3,2))
print(a)


c=torch.cat((a,b),1)
print(c)

print(c[0:-1,:])


a=np.arange(5)
b=torch.from_numpy(a)
print(a)
np.add(a,1,out=a)
print('\n')

print(a)
print('\n')
print(b)

b=a
print(b)
np.add(a,1,out=a)#这就是在原来的内存上直接加了一个1，而a=a+1是创建了一个新的内存。
#python就是这样，当你新创建一个变量的时候就创建一个内存。变量都是指向一个内存的指针

print(a)
print(b)


print(torch.cuda.is_available())




x=Variable(torch.ones(2,2),requires_grad=True)
print(x)
y=x+2
print(y)
print(y.creator)
z=y*y*3
print(z)
print(z.creator)

out=z.mean()
print(out)
out.backward()

print(x.grad)


x=torch.range(0,4,1)
x=Variable(x,requires_grad=True)
#x是一个变量，y也是一个变量，但是x是leaf。y是由一个乘法操作来的，

y=(x+2)*x

print(y,'\n')
print(y.creator)
print(y.data)
#范式norm

print(y)
gradients=torch.FloatTensor([1.0,1.0,0.0001,1.0,2.0])

y.backward(gradients,retain_variables=True)#这时候y是一个张量，求的是梯度。

print(x.grad)
