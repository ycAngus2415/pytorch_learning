import numpy as np
'''
this file is used to read the ckim data set, which has atribute id,label and data
'''
class FileRead(object):
    def __init__(self):
        self.train_data = {}
        #this is the atribute
        self.train_atribute = ['id','label','data']
    def readfile(self, filename='./data/data_sample.txt'):
        file = open(filename)
        j = 0
        for train in file.readlines():
            j +=1
            train_one = {}
            train_one['id']=train[0:7]
            train_one['label'] = train[ 8:11]
            data  = [float(x) for x in train[12:-1].split(' ') ]
            data = np.array(data)
            data = np.reshape(data, (15, 4, 101, 101))
            train_one['data'] = data
            self.train_data['train%d'%j] = train_one
        self.train_len = j
        file.close()

#ff = FileRead()
#ff.readfile()
#import matplotlib.pyplot as plt
#plt.figure()
#for i in range(15):
#    for j in range(4):
#        plt.subplot(15,4, i*4+j+1)
#        plt.imshow(ff.train_data['train1']['data'][i,j,:,:])
#plt.show()
#plt.imsave('./figure.png', ff.train_data['train1']['data'][1,1,:,:])
#print(ff.train_data['train1'])
