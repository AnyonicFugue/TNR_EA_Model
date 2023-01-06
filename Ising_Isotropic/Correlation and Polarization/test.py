import numpy as np
import cmath

K=0.5
ans=0
for theta in np.arange(0,2*np.pi,step=0.2):
    ans+=(cmath.sqrt((np.sinh(2*K)**2-cmath.exp(-(0+1j)*theta))/(np.sinh(2*K)**2-cmath.exp((0+1j)*theta))).real)
    #print(cmath.sqrt((np.sinh(2*K)**2-cmath.exp(-(0+1j)*theta))/(np.sinh(2*K)**2-cmath.exp((0+1j)*theta))).real)

print(ans/(10*np.pi))
#print(np.pi)
