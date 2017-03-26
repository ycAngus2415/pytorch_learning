import torch
import matplotlib.pyplot as plt
import numpy as np
x=plt.imread('figure_1.png')
y=x[:,:,0]
z=x[:,:,1]
kk=z-y
# print(kk)
# print(kk.size)
# print(kk.shape)
# print(960*1280)
zz=np.reshape(kk,(1,kk.size))
plt.imshow(z-y)
plt.show()
