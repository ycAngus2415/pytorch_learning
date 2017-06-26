import numpy as np

num = np.ones((3,3,3,2))
print(num)

min_num = 0
max_num = 2
print(min_num, max_num)
num = (num - min_num)/(max_num-min_num)
print(num)
