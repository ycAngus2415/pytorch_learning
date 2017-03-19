#this is my learning of PyTorch
ok .let's learning pytorn
##first is the tourial of deeplearning with pytorch for 60 minutes
###this is some basic operate of tensor in pytorch
###torch 和numpy 之间的转换
#autograd
##baceward
向后传播是标量向后传播，如果是tensor向后传播，必须要增加一个系数，这个系数的意思就是将这个tensor变成标量，按照一定的权值相加。
#现在开始学习pytorch的一些例子
##先用numpy 写了一个两层的神经网络，还是比较简单的。
##然后用torch的tensor写了一个两层的神经网络
记得学习率是一个非常重要的影响因素。什么都对了，就是因为学习率出现问题，导致乱码。气死了
##问题
现在我还是不能够使用atuograd.variable,和autograd.function简单的实现two_layer_net，可能是因为backward 对自己定义的forward 方法，其中线性自己写的话，就会出现不收敛的情况，只能说，backward具有一定的局限性。但是我现在还没有自己定义一个backward，我下次得试试。
#学习路线，就是按照。pytorch-example中的目录走的，还是比较详细。
在我这里。主要是numpy-torch-atuograd-function-tensorflow-nn-optim-module-dynamic_net
这条路线是真不错，不仅对比了torch和numpy的区别，还对别了其和tensorflow的区别，用他们分别作了实现。
#另外，在学习中，我发现relu其实是目前最好用的激活函数
#反向传播时，adam比sgd好用。
