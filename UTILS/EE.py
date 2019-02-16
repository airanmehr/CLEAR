import numpy as np
import pandas as pd

comaleName = r'\sc{Clear}'
class EE:
    @staticmethod
    def fx(x, s=0.0, h=0.5):
        Z=(1 + s) * x ** 2 + 2 * (1 + h * s) * x * (1 - x) + (1 - x) ** 2
        if Z>0:
            return ((1 + s) * x ** 2 + (1 + h * s) * x * (1 - x)) / (Z)
        else:
            return 0

    @staticmethod
    def sig(x): return 1. / (1 + np.exp(-x))

    @staticmethod
    def logit(p): return np.log(p) - np.log(1 - p)


    # def logit_(p): return T.log(p) - T.log(1 - p)


    # def sig_(x): return 1. / (1 + T.exp(-x))

    @staticmethod
    def Nu(s, t, nu0, theta, n=2000): return EE.Z(EE.sig(t * s / 2 + EE.logit(nu0)), n, theta)

    @staticmethod
    def forward(x0=0.005,h=0.5,s=1,t=150):
        def f(x,h=0.5,s=1): return ((1+s)*x*x + (1+h*s)*x*(1-x) )/((1+s)*x*x + 2*(1+h*s)*x*(1-x)  +(1-x)**2)
        x=[x0]
        for i in range(t):
            x+=[f(x[-1],h,s)]
        return pd.Series(x)

    floatX = 'float64'

    @staticmethod
    def Z(nu, n, theta): return theta * (
    nu * ((nu + 1) / 2. - 1. / ((1 - nu) * n + 1)) + (1 - nu) * ((n + 1.) / (2 * n) - 1. / ((1 - nu) * n + 1)))