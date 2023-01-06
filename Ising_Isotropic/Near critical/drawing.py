import matplotlib.pyplot as plt
import numpy as np

t=np.loadtxt('T.txt').ravel()
c=np.loadtxt('C.txt').ravel()/(t*t)
f=np.genfromtxt('F.txt').ravel()
e=np.loadtxt('E.txt').ravel()

plt.xlabel("kT")
plt.ylabel("Free energy")
#plt.xlim(0,810)

plt.plot(t,f)
plt.show()

plt.xlabel("kT")
plt.ylabel("Energy")
#plt.xlim(0,810)

plt.plot(t,-e)
plt.show()

plt.xlabel("kT")
plt.ylabel("Heat capacity")
#plt.xlim(0,810)
#plt.ylim(0,810)

plt.plot(t,c)
plt.grid(True,'both','both')
plt.show()


