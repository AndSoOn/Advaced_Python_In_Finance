#Embedded file name: C:\dev\numfin\master\lib\me.py
""" Maximum entropy objective function"""
import numpy as np

class MaxEntDual(object):

    def __init__(self, q, b, a, e):
        self.q, self.a, self.b, self.e = (q,
         a,
         b,
         e)

    def dual(self, x):
        return np.log(self.q.dot(np.exp(x.dot(self.b)))) - x.dot(self.a) + self.e.dot(x * x)

    def dist(self, x):
        p = self.q * np.exp(x.dot(self.b))
        return p / sum(p)

    def grad(self, x):
        p = self.dist(x)
        return self.b.dot(p) - self.a + 2 * self.e * x
