import matplotlib.pyplot as plt
import numpy as np

t=np.loadtxt('T.txt').ravel()
#c=np.loadtxt('C.txt').ravel()/(t*t)
Chi=np.genfromtxt('Chi.txt').ravel()
#e=np.loadtxt('E.txt').ravel()

plt.xlabel("kT")
plt.ylabel("Susceptibility")
plt.xlim(0,12)

plt.plot(t,Chi)
plt.show()


