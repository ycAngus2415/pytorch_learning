import tensorflow as tf
import numpy as np
import math
import time
def weight_init(shape):
      initial = tf.random_uniform(shape,minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]), maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
        return tf.Variable(initial,trainable=True)
# 全部初始化成0
def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial,trainable=True)
# 正交矩阵初始化
def orthogonal_initializer(shape,scale = 1.0):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    scale = 1.0
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape) #this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[0], :shape[1]],trainable=True, dtype=tf.float32)
def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)
# 洗牌
def shufflelists(data):
    ri=np.random.permutation(len(data))
    data=[data[i] for i in ri]
    return data


class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):

        # var
        # incoming是用来接收输入数据的，其形状为[n_samples, n_steps, D_cell]
        self.incoming = incoming
        # 输入的维度
        self.D_input = D_input
        # LSTM的hidden state的维度，同时也是memory cell的维度
        self.D_cell = D_cell
        # parameters
        # 输入门的 三个参数
        # igate = W_xi.* x + W_hi.* h + b_i
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i  = tf.Variable(tf.zeros([self.D_cell]))
        # 遗忘门的 三个参数
        # fgate = W_xf.* x + W_hf.* h + b_f
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f  = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # 输出门的 三个参数
        # ogate = W_xo.* x + W_ho.* h + b_o
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o  = tf.Variable(tf.zeros([self.D_cell]))
        # 计算新信息的三个参数
        # cell = W_xc.* x + W_hc.* h + b_c
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c  = tf.Variable(tf.zeros([self.D_cell]))
        # 如果没有特殊指定，这里直接设成全部为0
        init_for_both = tf.matmul(self.incoming[:,0,:], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        # 所以要将hidden state和memory并在一起。
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # 需要将数据由[n_samples, n_steps, D_cell]的形状变成[n_steps, n_samples, D_cell]的形状
        self.incoming = tf.transpose(self.incoming, perm=[1,0,2])


    def one_step(self, previous_h_c_tuple, current_x):

        # 再将hidden state和memory cell拆分开
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 这时，current_x是当前的输入，
        # prev_h是上一个时刻的hidden state
        # prev_c是上一个时刻的memory cell

        # 计算输入门
        i = tf.sigmoid(
            tf.matmul(current_x, self.W_xi) +
            tf.matmul(prev_h, self.W_hi) +
            self.b_i)
        # 计算遗忘门
        f = tf.sigmoid(
                    tf.matmul(current_x, self.W_xf) +
                    tf.matmul(prev_h, self.W_hf) +
                    self.b_f)
        # 计算输出门
        o = tf.sigmoid(
                    tf.matmul(current_x, self.W_xo) +
                    tf.matmul(prev_h, self.W_ho) +
                    self.b_o)
        # 计算新的数据来源
        c = tf.tanh(
                tf.matmul(current_x, self.W_xc) +
                tf.matmul(prev_h, self.W_hc) +
                self.b_c)
        # 计算当前时刻的memory cell
        current_c = f*prev_c + i*c
        # 计算当前时刻的hidden state
        current_h = o*tf.tanh(current_c)
        # 再次将当前的hidden state和memory cell并在一起返回
        return tf.stack([current_h, current_c])


    def all_steps(self):
        # 输出形状 : [n_steps, n_sample, D_cell]
        hstates = tf.scan(fn = self.one_step,
                          elems = self.incoming, #形状为[n_steps, n_sample, D_input]
                          initializer = self.previous_h_c_tuple,
                          name = 'hstates')[:,0,:,:]
        return hstates


D_input = 39
D_label = 24
learning_rate = 7e-5
num_units=1024
# 样本的输入和标签
inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
labels = tf.placeholder(tf.float32, [None, D_label], name="labels")
# 实例LSTM类
rnn_cell = LSTMcell(inputs, D_input, num_units, orthogonal_initializer)
# 调用scan计算所有hidden states
rnn0 = rnn_cell.all_steps()
# 将3维tensor [n_steps, n_samples, D_cell]转成 矩阵[n_steps*n_samples, D_cell]
# 用于计算outputs
rnn = tf.reshape(rnn0, [-1, num_units])
# 输出层的学习参数
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.matmul(rnn, W) + b
# 损失
loss=tf.reduce_mean((output-labels)**2)
# 训练所需
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# 建立session并实际初始化所有参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 训练并记录
def train_epoch(EPOCH):
    for k in range(EPOCH):
        train0=shufflelists(train)
        for i in range(len(train)):
            sess.run(train_step,feed_dict={inputs:train0[i][0],labels:train0[i][1]})
        tl=0
        dl=0
        for i in range(len(test)):
            dl+=sess.run(loss,feed_dict={inputs:test[i][0],labels:test[i][1]})
        for i in range(len(train)):
            tl+=sess.run(loss,feed_dict={inputs:train[i][0],labels:train[i][1]})
         print(k,'train:',round(tl/83,3),'test:',round(dl/20,3))


t0 = time.time()
train_epoch(10)
t1 = time.time()
print(" %f seconds" % round((t1 - t0),2))


