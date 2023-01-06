import cmath
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

t=np.loadtxt('T.txt').ravel()

m=np.loadtxt('M.txt').ravel()
#c=np.loadtxt('C.txt').ravel()/(t*t)
C01=np.genfromtxt('C01.txt').ravel()
C11=np.genfromtxt('C11.txt').ravel()*2
#e=np.loadtxt('E.txt').ravel()


plt.xlabel("kT")
plt.ylabel("C(0,1)")
plt.xlim(0,12)

plt.plot(t,C01)
plt.grid(True,'both','both')
x_sticks=np.arange(start=0,stop=12,step=0.5)
plt.xticks(x_sticks)

plt.show()



plt.xlabel("kT")
plt.ylabel("C(1,1)")
plt.xlim(0,12)

plt.plot(t,C11)
plt.grid(True,'both','both')
x_sticks=np.arange(start=0,stop=12,step=0.5)
plt.xticks(x_sticks)

plt.show()



n=np.size(t)
t1=[]
theory=[]
i=0
while t[i]<2.27:
    t1.append(t[i])
    i+=1


def integrand(theta,K,K1):
    return ( cmath.sqrt((K-cmath.exp(-(0+1j)*theta))/(K-cmath.exp((0+1j)*theta)))).real
    

for j in range(i):
    K=np.sinh(2/t1[j])**2
    theory.append(scipy.integrate.quad(integrand,0,2*np.pi,args=(K,0))[0]/(2*np.pi))

plt.xlabel("kT")
plt.ylabel("C(1,1)")

plt.plot(t,C11)
plt.plot(t1,theory)

plt.grid(True,'both','both')
x_sticks=np.arange(start=0,stop=10,step=0.5)
plt.xticks(x_sticks)

plt.show()
'''
plt.xlabel("kT")
plt.ylabel("Spontaneous magnetization")
plt.xlim(0,10)


Theory=np.zeros(np.size(t))
for i in range(np.size(t)):
    tmp=1/np.sinh(2/t[i])**4
    if(tmp>=1):
        Theory[i]=0
    else:
        Theory[i]=(1-tmp)**(1/8)

plt.plot(t,m)
plt.plot(t,Theory)


plt.grid(True,'both','both')
x_sticks=np.arange(start=0,stop=10,step=0.5)
plt.xticks(x_sticks)



plt.show()
'''
