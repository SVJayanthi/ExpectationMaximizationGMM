# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:27:08 2020

@author: srava
"""

from math import sqrt, log, exp, pi

class Gaussian:
    "Model Univariate Gaussian"
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, datum):
        return (1/sqrt(2*pi*self.sigma**2))*exp(-0.5*((datum-self.mu)/self.sigma)**2)
    
    def __repr__(self):
         return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)