# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:48:56 2020

@author: srava
"""


from math import sqrt, log
from random import uniform

from gaussian import Gaussian

class GMM: 
    "Gaussian Mixture Model"
    def __init__(self, data, sigma_min=1, sigma_max=1, mix=.5):
        mu_min=min(data)
        mu_max=max(data)
        self.data = data
        self.G1 = Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max))
        self.G2 = Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max))
        
        self.Prior1 = 0.5
        self.Prior2 = 0.5
        
        self.mix_factor = mix
        
    def E_step(self):
        "Expectation step"
        self.loglike = 0
        for datum in self.data:            
            pdf1 = self.G1.pdf(float(datum)) * self.Prior1
            pdf2 = self.G2.pdf(float(datum)) * self.Prior2
            
            tot_prob = pdf1 + pdf2
            pdf1 /= tot_prob   
            pdf2 /= tot_prob
            
            self.loglike += log(tot_prob) 
            
            yield (pdf1, pdf2)
        
    def M_step(self, weights):
        "Maximization step"
        
        (left, right) = zip(*weights) 
        sum_left = sum(left)
        sum_right = sum(right)
        
        self.G1.mu = sum(w * d  for (w, d) in zip(left, self.data)) / sum_left
        self.G2.mu = sum(w * d  for (w, d) in zip(right, self.data)) / sum_right
        
        self.G1.sigma = sqrt(sum(w * (d - self.G1.mu)**2  for (w, d) in zip(left, self.data)) / sum_left)
        self.G2.sigma = sqrt(sum(w * (d - self.G2.mu)**2  for (w, d) in zip(right, self.data)) / sum_right)
        
        self.Prior1 = sum_left / len(self.data)
        self.Prior2 = 1 - self.Prior1
        
    def iterate(self, N=1):
        "Iterate over N steps"
        for i in range(N):
            self.M_step(self.E_step())
        "Compute log-likelihood"
        self.E_step()
    
    def pdf(self, x):
        return (self.Prior1)*self.G1.pdf(x) + (self.Prior2)*self.G2.pdf(x)