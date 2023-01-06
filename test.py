import scipy
import scipy.sparse.linalg
import numpy as np
from matplotlib import pyplot as plt

rng=np.random.default_rng(132)
arr1=rng.random((4,5))*10

'''
arr1=np.ones((2,3))
res=open("result.txt",mode='a')
res.write('str(E_average)\n')
res.write(str(arr1))
res.close()
'''

print(np.geomspace(10,100,30))

