import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import torch.nn.functional as f
import torch
import skimage
from torch.autograd import Variable
'''

this file is used to read the ckim data set, which has atribute id,label and data
'''
class FileRead(object):
    def __init__(self):
        self.train_data = {}
        #this is the atribute
        self.train_atribute = ['id','label','data']
    def readfile(self, filename='./data/data_sample.txt'):
        try:
            file = open(filename)
            j = 0
            for train in file:
                j +=1
                train_one = {}
                train1 = train.split(',')
                train_one['id']=train1[0]
                train_one['label'] = float(train1[1])
                data  = [float(x) for x in train1[2].split(' ') ]
                data = np.array(data)
                data = np.reshape(data, (15, 4, 101, 101))
                train_one['data'] = data
                self.train_data[train_one['id']] = train_one
            self.train_len = j
        finally:
            file.close()
        return self.train_data
    def readprocess(self):

        with Pool(3) as p:
            s = p.map(self.readfile, ('./data/data_sampleaa','./data/data_sampleab'))
        for x in s :
            self.train_data.update(x)

        print(self.train_data)

ff = FileRead()
ff.readfile()
dd = []
figure = plt.figure(5)
torch.nn.Tanh
def close(data):
    ddd = 0
    for j in range(15):
        ddd +=data[j,:,:]
    return ddd
for i,data in enumerate(ff.train_data.values()):
    tt = f.max_pool2d(Variable(torch.FloatTensor(data['data'][:, 0, :, :])), (5, 5), (3, 3)).data.numpy()
    print(tt.shape)
    for j in range(15):
        plt.subplot(5, 15, i*15 +j+1)
        ee = tt[j]
        plt.imshow(ee)
plt.show()
#ff.readprocess()
#for x in s:
#    for f in x.values():
#        print(f['data'].shape)
#import matplotlib.pyplot as plt
#plt.figure()
#for i in range(15):
#    for j in range(4):
#        plt.subplot(15,4, i*4+j+1)
#        plt.imshow(ff.train_data['train1']['data'][i,j,:,:])
#plt.show()
#plt.imsave('./figure.png', ff.train_data['train1']['data'][1,1,:,:])
#print(ff.train_data['train1'])
