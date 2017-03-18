
# coding: utf-8

# # Introduction to PyTorch for former Torchies
# 
# In this tutorial, you will learn the following:
# 
# 1. Using torch Tensors, and important difference against (Lua)Torch
# 2. Using the autograd package
# 3. Building neural networks
#   - Building a ConvNet
#   - Building a Recurrent Net
#   - Using multiple GPUs
# 
# 
# ## Tensors 
# 
# Tensors behave almost exactly the same way in PyTorch as they do in Torch.

# In[ ]:

import torch
a = torch.FloatTensor(10, 20)
# creates tensor of size (10 x 20) with uninitialized memory

a = torch.randn(10, 20)
# initializes a tensor randomized with a normal distribution with mean=0, var=1

a.size()


# *NOTE: `torch.Size` is in fact a tuple, so it supports the same operations*
# 
# ### Inplace / Out-of-place
# 
# The first difference is that ALL operations on the tensor that operate in-place on it will have an `_` postfix.
# For example, `add` is the out-of-place version, and `add_` is the in-place version.

# In[ ]:

a.fill_(3.5)
# a has now been filled with the value 3.5

b = a.add(4.0)
# a is still filled with 3.5
# new tensor b is returned with values 3.5 + 4.0 = 7.5


# Some operations like `narrow` do not have in-place versions, and hence, `.narrow_` does not exist. 
# Similarly, some operations like `fill_` do not have an out-of-place version, so `.fill` does not exist.
# 
# ### Zero Indexing
# 
# Another difference is that Tensors are zero-indexed. (Torch tensors are one-indexed)

# In[ ]:

b = a[0,3] # select 1st row, 4th column from a


# Tensors can be also indexed with Python's slicing

# In[ ]:

b = a[:,3:5] # selects all rows, 4th column and  5th column from a


# ### No camel casing
# 
# The next small difference is that all functions are now NOT camelCase anymore.
# For example `indexAdd` is now called `index_add_`

# In[ ]:

x = torch.ones(5, 5)
print(x)


# In[ ]:

z = torch.Tensor(5, 2)
z[:,0] = 10
z[:,1] = 100
print(z)


# In[ ]:

x.index_add_(1, torch.LongTensor([4,0]), z)
print(x)


# ### Numpy Bridge
# 
# Converting a torch Tensor to a numpy array and vice versa is a breeze.
# The torch Tensor and numpy array will share their underlying memory locations, and changing one will change the other.
# 
# #### Converting torch Tensor to numpy Array

# In[ ]:

a = torch.ones(5)
a


# In[ ]:

b = a.numpy()
b


# In[ ]:

a.add_(1)
print(a)
print(b) # see how the numpy array changed in value


# #### Converting numpy Array to torch Tensor

# In[ ]:

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b) # see how changing the np array changed the torch Tensor automatically


# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.
# 
# ### CUDA Tensors
# 
# CUDA Tensors are nice and easy in pytorch, and they are much more consistent as well.
# Transfering a CUDA tensor from the CPU to GPU will retain it's type.

# In[ ]:

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    # creates a LongTensor and transfers it 
    # to GPU as torch.cuda.LongTensor
    a = torch.LongTensor(10).fill_(3).cuda()
    print(type(a))
    b = a.cpu()
    # transfers it to CPU, back to 
    # being a torch.LongTensor


# ## Autograd
# 
# Autograd is now a core torch package for automatic differentiation. 
# 
# It uses a tape based system for automatic differentiation. 
# 
# In the forward phase, the autograd tape will remember all the operations it executed, and in the backward phase, it will replay the operations.
# 
# In autograd, we introduce a `Variable` class, which is a very thin wrapper around a `Tensor`. 
# You can access the raw tensor through the `.data` attribute, and after computing the backward pass, a gradient w.r.t. this variable is accumulated into `.grad` attribute.
# 
# ![Variable](images/Variable.png)
# 
# There's one more class which is very important for autograd implementation - a `Function`. `Variable` and `Function` are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each variable has a `.creator` attribute that references a function that has created a function (except for Variables created by the user - these have `None` as  `.creator`).
# 
# If you want to compute the derivatives, you can call `.backward()` on a `Variable`. 
# If `Variable` is a scalar (i.e. it holds a one element tensor), you don't need to specify any arguments to `backward()`, however if it has more elements, you need to specify a `grad_output` argument that is a tensor of matching shape.

# In[ ]:

from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad = True)
x  # notice the "Variable containing" line


# In[ ]:

x.data


# In[ ]:

x.grad


# In[ ]:

x.creator is None  # we've created x ourselves


# In[ ]:

y = x + 2
y


# In[ ]:

y.creator
# y was created as a result of an operation, 
# so it has a creator


# In[ ]:

z = y * y * 3
z


# In[ ]:

out = z.mean()
out


# In[ ]:

# let's backprop now
out.backward()


# In[ ]:

# print gradients d(out)/dx
x.grad


# By default, gradient computation flushes all the internal buffers contained in the graph, so if you even want to do the backward on some part of the graph twice, you need to pass in `retain_variables = True` during the first pass.

# In[ ]:

x = Variable(torch.ones(2, 2), requires_grad = True)
y = x * x
y.backward(torch.ones(2, 2), retain_variables=True)
# the retain_variables flag will prevent the internal buffers from being freed
x.grad


# In[ ]:

# just backproping random gradients
gradient = torch.randn(2, 2)

# this would fail if we didn't specify 
# that we want to retain variables
y.backward(gradient)

x.grad


# ## nn package

# In[ ]:

import torch.nn as nn


# We've redesigned the nn package, so that it's fully integrated with autograd.
# 
# ### Replace containers with autograd
# 
# You no longer have to use Containers like ConcatTable, or modules like CAddTable, or use and debug with nngraph. 
# We will seamlessly use autograd to define our neural networks.
# For example, 
# 
# `output = nn.CAddTable():forward({input1, input2})` simply becomes `output = input1 + input2`
# 
# `output = nn.MulConstant(0.5):forward(input)` simply becomes `output = input * 0.5`
# 
# ### State is no longer held in the module, but in the network graph
# 
# Using recurrent networks should be simpler because of this reason. If you want to create a recurrent network, simply use the same Linear layer multiple times, without having to think about sharing weights.
# 
# ![torch-nn-vs-pytorch-nn](images/torch-nn-vs-pytorch-nn.png)
# 
# ### Simplified debugging
# 
# Debugging is intuitive using Python's pdb debugger, and **the debugger and stack traces stop at exactly where an error occurred.** What you see is what you get.
# 
# ### Example 1: ConvNet
# 
# Let's see how to create a small ConvNet. 
# 
# All of your networks are derived from the base class `nn.Module`.
# 
# - In the constructor, you declare all the layers you want to use.
# - In the forward function, you define how your model is going to be run, from input to output

# In[ ]:

import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    def __init__(self):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)
        
    # it's the forward function that defines the network structure
    # we're accepting only a single input in here, but if you want,
    # feel free to use more
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))

        # in your model definition you can go full crazy and use arbitrary
        # python code to define your model structure
        # all these are perfectly legal, and will be handled correctly 
        # by autograd:
        # if x.gt(0) > x.numel() / 2:
        #      ...
        # 
        # you can even do a loop and reuse the same module inside it
        # modules no longer hold ephemeral state, so you can use them
        # multiple times during your forward pass        
        # while x.norm(2) < 10:
        #    x = self.conv1(x) 
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# Let's use the defined ConvNet now.  
# You create an instance of the class first.

# In[ ]:

net = MNISTConvNet()
print(net)


# > #### NOTE: `torch.nn` only supports mini-batches
# The entire `torch.nn` package only supports inputs that are a mini-batch of samples, and not a single sample.  
# For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.
# 
# > *If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.*

# In[ ]:

# create a mini-batch containing a single sample of random data
input = Variable(torch.randn(1, 1, 28, 28))

# send the sample through the ConvNet
out = net(input)
print(out.size())


# In[ ]:

# define a dummy target label
target = Variable(torch.LongTensor([3]))

# create a loss function
loss_fn = nn.CrossEntropyLoss() # LogSoftmax + ClassNLL Loss
err = loss_fn(out, target)
print(err)


# In[ ]:

err.backward()


# The output of the ConvNet `out` is a `Variable`. We compute the loss using that, and that results in `err` which is also a `Variable`.
# 
# Calling `.backward` on `err` hence will propagate gradients all the way through the ConvNet to it's weights

# ##### Let's access individual layer weights and gradients

# In[ ]:

print(net.conv1.weight.grad.size())


# In[ ]:

print(net.conv1.weight.data.norm()) # norm of the weight
print(net.conv1.weight.grad.data.norm()) # norm of the gradients


# ### Forward and Backward Function Hooks
# We've inspected the weights and the gradients. 
# But how about inspecting / modifying the output and grad_output of a layer?
# 
# We introduce **hooks** for this purpose.
# 
# You can register a function on a *Module* or a *Variable*.  
# The hook can be a forward hook or a backward hook.  
# The forward hook will be executed when a forward call is executed.  
# The backward hook will be executed in the backward phase.  
# Let's look at an example.
# 

# In[ ]:

# We register a forward hook on conv2 and print some information
def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

net.conv2.register_forward_hook(printnorm)

out = net(input)


# In[ ]:

# We register a backward hook on conv2 and print some information
def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')    
    print('Inside class:' + self.__class__.__name__)
    print('')    
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')    
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())

net.conv2.register_backward_hook(printgradnorm)

out = net(input)
err = loss_fn(out, target)
err.backward()


# A full and working MNIST example is located here
# https://github.com/pytorch/examples/tree/master/mnist
# 
# ### Example 2: Recurrent Net
# 
# Next, let's lookm at building recurrent nets with PyTorch.
# 
# Since the state of the network is held in the graph and not
# in the layers, you can simply create an nn.Linear and 
# reuse it over and over again for the recurrence.

# In[ ]:

class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        input_size = data_size + hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output

rnn = RNN(50, 20, 10)


# In[ ]:

loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = Variable(torch.randn(batch_size, 50))
hidden = Variable(torch.zeros(batch_size, 20))
target = Variable(torch.zeros(batch_size, 10))

loss = 0
for t in range(TIMESTEPS):                  
    # yes! you can reuse the same network several times,
    # sum up the losses, and call backward!
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()


# A more complete Language Modeling example using LSTMs and Penn Tree-bank is located here: https://github.com/pytorch/examples/tree/master/word_language_model
# 
# PyTorch by default has seamless CuDNN integration for ConvNets and Recurrent Nets

# ### Multi-GPU examples
# 
# Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches and run the computation for each of the smaller mini-batches in parallel.
# 
# Data Parallelism is implemented using `torch.nn.DataParallel`.
# 
# One can wrap a Module in `DataParallel` and it will be parallelized over multiple GPUs in the batch dimension.

# #### Data Parallel

# In[ ]:

class DataParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Linear(10, 20)
        
        # wrap block2 in DataParallel
        self.block2=nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)
        
        self.block3=nn.Linear(20, 20)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# The code does not need to be changed in CPU-mode.
# 
# [The documentation for DataParallel is here](http://pytorch.org/docs/nn.html#torch.nn.DataParallel)

# #### Primitives on which data parallel is implemented upon
# In general, pytorch's nn.parallel primitives can be used independently.
# We have implemented simple MPI-like primitives:
# - replicate: replicate a Module on multiple devices
# - scatter: distribute the input in the first-dimension
# - gather: gather and concatenate the input in the first-dimension
# - parallel_apply: apply a set of already-distributed inputs to a set of already-distributed models.
# 
# To give a better clarity, here function `data_parallel` composed using these collectives

# In[ ]:

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


# #### Part of the model on CPU and part on the GPU
# 
# Let's look at a small example of implementing a network where part of it is on the CPU and part on the GPU
# 

# In[ ]:

class DistributedModel(nn.Module):
    def __init__(self):
        super().__init__(
            embedding=nn.Embedding(1000, 10),
            rnn=nn.Linear(10, 10).cuda(0),
        )
        
    def forward(self, x):
        # Compute embedding on CPU
        x = self.embedding(x)
        
        # Transfer to GPU
        x = x.cuda(0)
        
        # Compute RNN on GPU
        x = self.rnn(x)
        return x


# This was a small introduction to PyTorch for former Torch users.
# 
# There's a lot more to learn.
# 
# Look at our more comprehensive introductory tutorial which introduces the `optim` package, data loaders etc.: [Deep Learning with PyTorch: a 60-minute blitz](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb)
# 
# Also look at 
# 
# - [Train neural nets to play video games](https://goo.gl/uGOksc)
# - [Train a state-of-the-art ResNet network on imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
# - [Train an face generator using Generative Adversarial Networks](https://github.com/pytorch/examples/tree/master/dcgan)
# - [Train a word-level language model using Recurrent LSTM networks](https://github.com/pytorch/examples/tree/master/word_language_model)
# - [More examples](https://github.com/pytorch/examples)
# - [More tutorials](https://github.com/pytorch/tutorials)
# - [Discuss PyTorch on the Forums](https://discuss.pytorch.org/)
# - [Chat with other users on Slack](pytorch.slack.com/messages/beginner/)

# In[ ]:



