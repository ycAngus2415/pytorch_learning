<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
this is my learning of PyTorch  

**ok .let's learning pytorn**
##first is the tourial of deeplearning with pytorch for 60 minutes  
###this is some basic operate of tensor in py torch  
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
=======
# this is my learning of PyTorch  
ok .let's learning pytorn
## first is the tourial of deeplearning with pytorch for 60 minutes
### this is some basic operate of tensor in pytorch
###  torch 和numpy 之间的转换
#autograd
=======
# this is my learning of PyTorch  
ok .let's learning pytorn
## first is the tourial of deeplearning with pytorch for 60 minutes
### this is some basic operate of tensor in pytorch
###  torch 和numpy 之间的转换
#autograd
>>>>>>> origin/master
=======
# this is my learning of PyTorch  
ok .let's learning pytorn
## first is the tourial of deeplearning with pytorch for 60 minutes
### this is some basic operate of tensor in pytorch
###  torch 和numpy 之间的转换
#autograd
>>>>>>> origin/master
##  baceward
向后传播是标量向后传播，如果是tensor向后传播，必须要增加一个系数，这个系数的意思就是将这个tensor变成标量，按照一定的权值相加。
#现在开始学习pytorch的一些例子
## 先用numpy 写了一个两层的神经网络，还是比较简单的。
## 然后用torch的tensor写了一个两层的神经网络
记得学习率是一个非常重要的影响因素。什么都对了，就是因为学习率出现问题，导致乱码。气死了
## 问题
现在我还是不能够使用atuograd.variable,和autograd.function简单的实现two_layer_net，可能是因为backward 对自己定义的forward 方法，其中线性自己写的话，就会出现不收敛的情况，只能说，backward具有一定的局限性。但是我现在还没有自己定义一个backward，我下次得试试。
# 学习路线，就是按照。pytorch-example中的目录走的，还是比较详细。
在我这里。主要是**numpy-torch-atuograd-function-tensorflow-nn-optim-module-dynamic_net**
这条路线是真不错，不仅对比了torch和numpy的区别，还对别了其和tensorflow的区别，用他们分别作了实现。
# 另外，在学习中，我发现relu其实是目前最好用的激活函数
# 反向传播时，adam比sgd好用。
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/master
=======
>>>>>>> origin/master
=======
>>>>>>> origin/master

# 写了一个卷积循环神经网络。

## 三层的LSTM
* 每一层都是一个时间周期个LSTM Cell
* 每一个Cell 内部都是卷积和直积。
* 输入有一个卷积， hidden 有一个卷积。cell是直积得到的。
* 每一层都可以得到一个输出，输出是一个值。不管是多少个hidden state， 都直接卷积到一维数。
* 最终结果是三层卷积循环神经网络的结果 再把结果做一次卷积的结果。
* 输入是一张三维图，但是我们还要加一维的通道，还要加一维的时间，还要加一维的batch
* **关键**torch 中都是有一个batchsize的，也就是，每一次输入都不是一个单纯的值，它需要多个数据输入，就算是一个数据，也要指明这个表示数据量的维数，不能舍弃。

## 经验总结
* 重要的一点，写程序前，脑袋里面一定要对流程熟悉，不熟悉的话就多画画
* 另外，对于网络中的数据结构一定要清楚，每一步之后的数据结构要清楚，最好在写代码之前在草稿上算好。
* 最后，要学会调试啊，没有用什么很高级的技术，反正我就是哪有问题，一步步缩小问题的范围，数据结构错了就看看size,反正多看看中间结果是没错的。
