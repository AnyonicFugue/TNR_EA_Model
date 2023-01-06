import sympy
import math
import scipy.integrate as inte
import matplotlib.pyplot as plt
import numpy as np
J = 1


def f(y, x, beta, aux):
    res = math.log((math.cosh(2*beta*J))**2 - math.sinh(2*beta*J)
                   * math.cos(y)-math.sinh(2*beta*J)*math.cos(x))
    return res


def calc_beta(beta):
    integral = inte.dblquad(f, 0, 2*math.pi, lambda x: 0,
                            lambda x: 2*math.pi, (beta, 0))
    print(integral)
    res = -(math.log(2)+1/(8*math.pi**2)*integral[0])/beta
    return res


kT_array = np.loadtxt('T.txt').ravel()
f_array = np.array([calc_beta(1/b) for b in kT_array])*2048
f_tn = np.genfromtxt('F.txt').ravel()
error = (f_tn-f_array)/f_array

txt = open("theoretical.txt", mode='a')
txt.write(str(f_array))
txt.close()

txt = open("error.txt", mode='a')
txt.write(str(error))
txt.close()

plt.plot(kT_array, error)
plt.xlabel("kT")
plt.ylabel("Relative error")
plt.show()
