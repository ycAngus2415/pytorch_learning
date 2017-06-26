import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plot
#for j in np.arange(4,15,1):
#    sum_e = 0
#    for i in range(10000):
#        x = np.random.uniform(0,100,4)
#        y = np.random.uniform(0,100,4)
#        x_ = stats.uniform(0,100).pdf(x)
#        y_ = stats.uniform(0,100).pdf(x)
#        x_y_ = np.multiply(x_, y_)
#        xy = np.abs(y-x)
#        e = np.sum(np.multiply(x_y_, xy))
#        sum_e += e
#    print(sum_e/(10000))
e = []
plot.figure()
plot.subplot(3,1,1)
sum = []
kk = []
for i in range(100000):
    x = np.random.normal(0,1, 100)
    x_ = stats.norm(0,1).pdf(x)

    plot.plot(x, x_, 'o')
    k = np.sum(np.multiply(x, x_))
    kk.append(k)
    o = k/np.sum(x_)
    sum.append(np.sum(kk))
    e.append(o)
plot.subplot(3,1,2)
plot.plot(range(100000), e)
plot.plot(range(100000),np.zeros(100000))
plot.subplot(3,1,3)
plot.plot(range(100000), sum)
plot.plot(range(100000), np.zeros(100000))
plot.show()

