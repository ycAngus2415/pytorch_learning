import numpy as np
import time
zz = np.random.normal(size=(3, 4, 5))
zz = np.reshape(zz, (3, 20))
print(zz)
print(zz.shape)
t2 = time.time()
np.savetxt('./tss', zz)
print(time.time() - t2)
cc = np.loadtxt('./tss')
print(cc)
print(cc.shape)
