import numpy as np
from readfile_ckim import FileRead
ff = FileRead()
ff.readfile()
x = []
y = []
for value in ff.train_data.values():

    x.append(value['data'])
    y.append(float(value['label']))
print(len(x))
print(len(y))

